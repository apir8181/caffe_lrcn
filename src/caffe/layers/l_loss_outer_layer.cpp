
#include "caffe/l_loss_outer_layer.hpp"

namespace caffe {

template <typename Dtype>
void LLossOuterLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
    << "data & data label should have equal shape.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
    << "data & data label should have equal shape.";

  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void LLossOuterLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  clip_size_ = bottom[0]->shape(0);
  batch_size_ = bottom[0]->shape(1);
  feature_size_ = bottom[0]->shape(2);
  instance_size_ = clip_size_ * batch_size_;

  vector<int> frame_pair_shape(2, instance_size_);
  S_frame_.Reshape(frame_pair_shape);
  I_frame_.Reshape(frame_pair_shape);
  pairwise_error_.Reshape(frame_pair_shape);
  pairwise_loss_.Reshape(frame_pair_shape);

}

#ifdef CPU_ONLY
STUB_GPU(LLossOuterLayer);
#endif

INSTANTIATE_CLASS(LLossOuterLayer);
REGISTER_LAYER_CLASS(LLossOuter);

}
