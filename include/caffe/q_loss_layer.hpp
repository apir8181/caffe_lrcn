#ifndef CAFFE_Q_LOSS_LAYERS_HPP_
#define CAFFE_Q_LOSS_LAYERS_HPP_

#include "caffe/blob.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
class QLossLayer : public LossLayer<Dtype> {
public:
  explicit QLossLayer(const LayerParameter &param)
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
        
  virtual inline const char* type() const { return "QLoss"; } 
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
        
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);
private:
  Dtype margin_width_;
  Dtype lambda_;
  Blob<Dtype> loss_; // same shape as bottom
};

} // namespace caffe

#endif // CAFFE_Q_LOSS_LAYERS_HPP_
