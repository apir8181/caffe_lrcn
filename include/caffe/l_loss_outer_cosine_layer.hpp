#ifndef CAFFE_L_LOSS_OUTER_COSINE_LAYER_HPP_
#define CAFFE_L_LOSS_OUTER_COSINE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
class LLossOuterCosineLayer : public LossLayer<Dtype> {
public:
  explicit LLossOuterCosineLayer(const LayerParameter& param) 
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
        
  virtual inline const char* type() const { return "LLossOuterCosine"; } 
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
        
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);

private:
  int clip_size_, batch_size_, feature_size_, instance_size_;
  bool normalize_;
  Blob<Dtype> L_frame_;
  Blob<Dtype> S_frame_;
  Blob<Dtype> I_frame_;
  Blob<Dtype> pairwise_error_;
  Blob<Dtype> pairwise_loss_;
};
  
} // namespace caffe

#endif // CAFFE_L_LOSS_LAYERS_HPP_
