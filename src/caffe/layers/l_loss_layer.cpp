
#include "caffe/l_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void LLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
    << "data & data label should have equal shape.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
    << "data & data label should have equal shape.";

  margin_width_ = this->layer_param_.l_loss_param().margin_width();
  normalize_ = this->layer_param_.loss_param().normalize();

  // set up bias term
  Dtype bias_init = this->layer_param_.l_loss_param().bias_init();
  this->blobs_.resize(1);
  vector<int> bias_shape(1, 1);
  this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
  caffe_gpu_set<Dtype>(1, bias_init, this->blobs_[0]->mutable_gpu_data());
}

template <typename Dtype>
void LLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  clip_size_ = bottom[0]->shape(0);
  batch_size_ = bottom[0]->shape(1);
  feature_size_ = bottom[0]->shape(2);
  instance_size_ = clip_size_ * batch_size_;

  vector<int> length_shape(1, instance_size_);
  L_frame_.Reshape(length_shape);

  vector<int> frame_pair_shape(2, instance_size_);
  vector<int> video_pair_shape(2, batch_size_);

  S_frame_.Reshape(frame_pair_shape);
  S_video_.Reshape(video_pair_shape);
  I_video_.Reshape(video_pair_shape);
  pairwise_loss_.Reshape(video_pair_shape);
  accuracy_.Reshape(video_pair_shape);
}

template <typename Dtype>
void LLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(LLossLayer);
#endif

INSTANTIATE_CLASS(LLossLayer);
REGISTER_LAYER_CLASS(LLoss);

}
