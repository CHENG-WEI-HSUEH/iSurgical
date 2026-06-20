#include "lc_seg_postprocessor_gpu.hpp"
#include <holoscan/core/gxf/gxf_utils.hpp>
#include <nvtx3/nvToolsExt.h>

void LcSegPostprocesserGPUOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<nvidia::gxf::Entity>("in");
    spec.output<nvidia::gxf::Entity>("out");

    spec.param(this->Mallocator, "MemoryAllocator", "Memory pool allocator");
    spec.param(this->colour_labels_, "colour_labels", "Colour labels and structural configs");
}

void LcSegPostprocesserGPUOp::start()
{
    const YAML::Node& colour_labels = this->colour_labels_.get();
    this->label_names_ = colour_labels["label_names"].as<std::vector<std::string>>();
    this->kclass_colours_ = colour_labels["kclass_colours"].as<std::vector<std::vector<float>>>();
    this->mask_threshold_ = colour_labels["MaskThreshold"].as<float>();
    this->height_ = colour_labels["Sys_Height"].as<int>();
    this->width_ = colour_labels["Sys_Width"].as<int>();
    this->inference_resize_width_ = colour_labels["InferenceResizeWidth"].as<int>();
    this->inference_resize_height_ = colour_labels["InferenceResizeHeight"].as<int>();
    this->num_blocks_ = colour_labels["NumBlocks"].as<int>(); // 4 channels (RGBA)

    // Parse head groups dynamically mapping to model.py structure
    if (colour_labels["head_groups"]) {
        for (const auto& group : colour_labels["head_groups"]) {
            this->head_groups_.push_back(group.as<std::vector<int>>());
        }
    }
    this->num_heads_ = static_cast<int>(this->head_groups_.size());

    cudaStreamCreate(&this->cuda_stream);

    // Upload colors directly to GPU constant space
    {
        int num_classes = static_cast<int>(this->kclass_colours_.size());
        std::vector<float> flat_colours(num_classes * 4, 1.0f);
        for (int c = 0; c < num_classes; ++c) {
            const auto& colour = this->kclass_colours_[c];
            for (size_t ch = 0; ch < colour.size() && ch < 4; ++ch) {
                flat_colours[c * 4 + ch] = colour[ch];
            }
        }
        UploadLcClassColorsToGPU(flat_colours.data(), num_classes);
    }

    // CPU backdrop drawing allocations for legend card
    cudaMallocHost(&this->h_box_roi_, this->width_ * this->height_ * this->num_blocks_ * sizeof(float));
    this->BoxRoiMat = cv::Mat(this->height_, this->width_, CV_32FC4, this->h_box_roi_);
    this->BoxRoiMat.setTo(cv::Scalar(0,0,0,0));

    this->OverlayInitialised = false;
}

void LcSegPostprocesserGPUOp::stop()
{
    if (this->cuda_stream) {
        cudaStreamSynchronize(this->cuda_stream);
    }

    this->OverlayEntity = nvidia::gxf::Entity();
    this->OverlayInitialised = false;
    this->d_overlay = nullptr;

    if (this->h_box_roi_) { cudaFreeHost(this->h_box_roi_); this->h_box_roi_ = nullptr; }

    if (this->cuda_stream) {
        cudaStreamDestroy(this->cuda_stream);
        this->cuda_stream = nullptr;
    }
}

void LcSegPostprocesserGPUOp::ReceiveInputs(holoscan::InputContext& op_input,
                                           std::vector<std::shared_ptr<holoscan::Tensor>>& head_tensors)
{
    auto InMsg = op_input.receive<holoscan::TensorMap>("in");
    auto& InTensors = InMsg.value();

    head_tensors.clear();
    // Expecting names mapped sequentially matching multi-head output index: "head_out_0", "head_out_1", etc.
    for (int idx = 0; idx < this->num_heads_; ++idx) {
        std::string head_name = "head_out_" + std::to_string(idx);
        if (InTensors.find(head_name) != InTensors.end()) {
            head_tensors.push_back(InTensors[head_name]);
        } else {
            throw std::runtime_error("Missing expected multi-head tensor output: " + head_name);
        }
    }
}

void LcSegPostprocesserGPUOp::ClearOverlay()
{
    if (this->d_overlay != nullptr) {
        size_t total_bytes = this->width_ * this->height_ * this->num_blocks_ * sizeof(float);
        cudaMemsetAsync(this->d_overlay, 0, total_bytes, this->cuda_stream);
    }
}

nvidia::gxf::Entity LcSegPostprocesserGPUOp::InitOverlayTensor(
                                            holoscan::ExecutionContext& context,
                                            std::shared_ptr<holoscan::Allocator> allocator,
                                            const std::string& name)
{
    auto entity = nvidia::gxf::Entity::New(context.context());
    auto tensor = entity.value().add<nvidia::gxf::Tensor>(name.c_str());
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator->gxf_cid());

    nvidia::gxf::Shape shape{this->height_, this->width_, this->num_blocks_};

    tensor.value()->reshape<float>(
        shape,
        nvidia::gxf::MemoryStorageType::kDevice,
        gxf_allocator.value()
    );

    this->d_overlay = tensor.value()->data<float>().value();
    cudaMemset(this->d_overlay, 0, this->width_ * this->height_ * this->num_blocks_ * sizeof(float));

    return holoscan::gxf::Entity(std::move(entity.value()));
}

void LcSegPostprocesserGPUOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context)
{
    nvtxRangePush("LcSegPostprocesserGPUOp::compute");

    if (!this->OverlayInitialised) {
        this->OverlayEntity = InitOverlayTensor(context, this->Mallocator, "OverlayTensor");
        this->OverlayInitialised = true;
    }

    ClearOverlay();

    std::vector<std::shared_ptr<holoscan::Tensor>> head_tensors;
    ReceiveInputs(op_input, head_tensors);

    // Render each head sequentially; later heads overwrite earlier ones on overlapping
    // pixels, matching the reference decode order in inference_video.py.
    for (int h = 0; h < this->num_heads_; ++h) {
        const float* d_head_logits = static_cast<const float*>(head_tensors[h]->data());
        int num_channels = static_cast<int>(this->head_groups_[h].size()) + 1; // channel 0 = background

        // channel_class_map[channel] = global class id. Channel 0 is the head background;
        // channels 1..N map to head_groups_[h][0..N-1] (see model.py: Conv2d(.., len(group)+1)
        // and inference_video.py: pred_mask[h_pred == local_i + 1] = group[local_i]).
        std::vector<int> channel_class_map(num_channels, 0);
        for (size_t c = 0; c < this->head_groups_[h].size(); ++c) {
            channel_class_map[c + 1] = this->head_groups_[h][c];
        }

        int* d_mapping = nullptr;
        cudaMalloc(&d_mapping, num_channels * sizeof(int));
        cudaMemcpy(d_mapping, channel_class_map.data(), num_channels * sizeof(int), cudaMemcpyHostToDevice);

        RenderLcHeadsToOverlayGPU(
            d_head_logits,
            d_mapping,
            num_channels,
            this->d_overlay,
            this->width_,
            this->height_,
            this->inference_resize_width_,
            this->inference_resize_height_,
            this->cuda_stream
        );

        cudaFree(d_mapping);
    }

    // Draw the legend last and blit ONLY its panel rectangle on top of the masks, so the
    // legend is never covered by a segmentation mask. A full-frame copy here would instead
    // overwrite the just-rendered masks everywhere outside the panel with transparent pixels.
    this->BoxRoiMat.setTo(cv::Scalar(0, 0, 0, 0));
    cv::Rect panel;
    try {
        panel = DrawLegendOnOverlay();
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("DrawLegendOnOverlay caught exception: {}", e.what());
        panel = cv::Rect();
    }

    if (panel.width > 0 && panel.height > 0) {
        size_t row_pitch = this->width_ * this->num_blocks_ * sizeof(float);
        const float* h_src = this->h_box_roi_ + (panel.y * this->width_ + panel.x) * this->num_blocks_;
        float* d_dst = this->d_overlay + (panel.y * this->width_ + panel.x) * this->num_blocks_;
        cudaMemcpy2DAsync(
            d_dst,
            row_pitch,                                              // dst pitch (full overlay width)
            h_src,
            row_pitch,                                              // src pitch (full BoxRoiMat width)
            panel.width * this->num_blocks_ * sizeof(float),        // copy only the panel width
            panel.height,
            cudaMemcpyHostToDevice,
            this->cuda_stream
        );
    }

    op_output.emit(this->OverlayEntity, "out");
    nvtxRangePop();
}

cv::Rect LcSegPostprocesserGPUOp::DrawLegendOnOverlay() {
    std::vector<int> active_classes;
    for (int cls_id = 1; cls_id < static_cast<int>(this->label_names_.size()); ++cls_id) {
        active_classes.push_back(cls_id);
    }
    if (active_classes.empty()) return cv::Rect();

    int item_height = 30;
    int padding = 15;
    int color_box_size = 20;
    int columns = 2;
    int col_width = 290;
    int col_gap = 20;

    int num_items = static_cast<int>(active_classes.size());
    int rows = (num_items + columns - 1) / columns;
    int legend_w = (padding * 2) + (col_width * columns) + (col_gap * (columns - 1));
    int legend_h = (rows * item_height) + (padding * 2);

    int margin = 20;
    int start_x = this->width_ - legend_w - margin;
    int start_y = margin;

    // Clamp to the frame so both the OpenCV ROI and the caller's blit stay in bounds.
    cv::Rect panel_rect = cv::Rect(start_x, start_y, legend_w, legend_h) &
                          cv::Rect(0, 0, this->width_, this->height_);
    if (panel_rect.width <= 0 || panel_rect.height <= 0) return cv::Rect();

    cv::Mat panel_roi = this->BoxRoiMat(panel_rect);
    cv::Mat color_card(panel_roi.size(), panel_roi.type(), cv::Scalar(0.0f, 0.0f, 0.0f, 0.6f));
    cv::addWeighted(color_card, 0.6, panel_roi, 0.4, 0.0, panel_roi);

    for (int current_item = 0; current_item < num_items; ++current_item) {
        int cls_id = active_classes[current_item];
        int row = current_item / columns;
        int col = current_item % columns;

        int col_x = start_x + padding + col * (col_width + col_gap);
        int row_y = start_y + padding + (row * item_height);

        cv::Scalar swatch_color(
            this->kclass_colours_[cls_id][0],
            this->kclass_colours_[cls_id][1],
            this->kclass_colours_[cls_id][2],
            1.0f
        );

        cv::Rect color_box(col_x, row_y + 4, color_box_size, color_box_size);
        cv::rectangle(this->BoxRoiMat, color_box, swatch_color, cv::FILLED);
        cv::rectangle(this->BoxRoiMat, color_box, cv::Scalar(1.0f, 1.0f, 1.0f, 1.0f), 1); 

        std::string display_name = this->label_names_[cls_id];
        cv::Point text_pos(col_x + color_box_size + 12, row_y + 20);
        cv::putText(
            this->BoxRoiMat,
            display_name,
            text_pos,
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(1.0f, 1.0f, 1.0f, 1.0f),
            1,
            cv::LINE_AA
        );
    }

    return panel_rect;
}

LcSegPostprocesserGPUOp::~LcSegPostprocesserGPUOp()
{
    if (this->h_box_roi_) cudaFreeHost(this->h_box_roi_);
}
