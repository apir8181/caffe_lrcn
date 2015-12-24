
#include "caffe/l_loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaxMarginAvgSimVideoLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
    << "data & data label should have equal shape.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
    << "data & data label should have equal shape.";

  margin_width_ = this->layer_param_.l_loss_param().margin_width();
  clip_size_ = bottom[0]->shape(0);
  batch_size_ = bottom[0]->shape(1);
  feature_size_ = bottom[0]->shape(2);
  normalize_ = this->layer_param_.loss_param().normalize();

  // set up bias term
  this->blobs_.resize(1);
  vector<int> bias_shape(1, 1);
  this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
  caffe_gpu_set<Dtype>(1, Dtype(0), this->blobs_[0]->mutable_gpu_data());
}

template <typename Dtype>
void MaxMarginAvgSimVideoLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  vector<int> video_pair_shape(2, batch_size_);
  S_video_.Reshape(video_pair_shape);
  I_video_.Reshape(video_pair_shape);
  pairwise_loss_.Reshape(video_pair_shape);

  vector<int> centroid_shape;
  centroid_shape.push_back(batch_size_);
  centroid_shape.push_back(feature_size_);
  video_centroid_.Reshape(centroid_shape);
}

template <typename Dtype>
void MaxMarginAvgSimVideoLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MaxMarginAvgSimVideoLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MaxMarginAvgSimVideoLossLayer);
#endif

INSTANTIATE_CLASS(MaxMarginAvgSimVideoLossLayer);
REGISTER_LAYER_CLASS(MaxMarginAvgSimVideoLoss);

}
