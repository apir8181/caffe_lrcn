
#include "caffe/l_loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaxMarginMaxSimVideoLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
    << "data & data label should have equal shape.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
    << "data & data label should have equal shape.";

  margin_width_ = this->layer_param_.max_margin_max_sim_video_loss_param().margin_width();
  clip_size_ = bottom[0]->shape(0);
  batch_size_ = bottom[0]->shape(1);
  feature_size_ = bottom[0]->shape(2);
  instance_size_ = clip_size_ * batch_size_;
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void MaxMarginMaxSimVideoLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  vector<int> frame_pair_shape;
  frame_pair_shape.push_back(instance_size_);
  frame_pair_shape.push_back(instance_size_);
  frame_pair_shape.push_back(1);
  frame_pair_shape.push_back(1);
  
  vector<int> video_pair_shape;
  video_pair_shape.push_back(batch_size_);
  video_pair_shape.push_back(batch_size_);
  video_pair_shape.push_back(1);
  video_pair_shape.push_back(1);

  S_frame_.Reshape(frame_pair_shape);
  I_frame_.Reshape(frame_pair_shape);
  N_frame_.Reshape(frame_pair_shape);
  S_video_.Reshape(video_pair_shape);
  I_video_.Reshape(video_pair_shape);
  pairwise_loss_.Reshape(video_pair_shape);
  temp_frame_.Reshape(frame_pair_shape);
}

template <typename Dtype>
void MaxMarginMaxSimVideoLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MaxMarginMaxSimVideoLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MaxMarginMaxSimVideoLossLayer);
#endif

INSTANTIATE_CLASS(MaxMarginMaxSimVideoLossLayer);
REGISTER_LAYER_CLASS(MaxMarginMaxSimVideoLoss);

}
