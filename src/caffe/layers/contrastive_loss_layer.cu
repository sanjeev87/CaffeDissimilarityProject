#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/* original code commented out 
template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}
*/
template <typename Dtype>
__global__ void CLLForward(const int count, const int channels,
    const Dtype margin, const Dtype alpha,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      if ((margin-dist_sq[n]) > 0.0) {
        bottom_diff[i] = -alpha * diff[i];
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}


template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  /*
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  */

  /*
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  */

  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  //margin = Dtype(1000);
  Dtype loss(0.0);


     printf("CLL_CU : the values of a_i are \n");
    for (int i = 0; i < bottom[0]->num(); ++i) {
       for (int j = 0; j < channels; ++j) {
          printf("%f \t ",(float) bottom[0]->cpu_data()[i*channels+j] );
      }
    }
   printf("CLL_CU : End printing values of a_i\n");

  printf("CLL_CU : the values of b_i are \n");
    for (int i = 0; i < bottom[1]->num(); ++i) {
       for (int j = 0; j < channels; ++j) {
          printf("%f \t ",(float) bottom[1]->cpu_data()[i*channels+j] );
      }
    }
   printf("CLL_CU : End printing values of b_i\n");

   printf("CLL_CU : the diff values for the input vector are \n");
   for(int temp = 0 ; temp < count ; temp++){
    printf("%f \t ",(float) diff_.mutable_cpu_data()[temp] );
   }
   printf("CLL_CU : End printing the diff values\n");


  for (int i = 0; i < bottom[0]->num(); ++i) {

  dist_sq_.mutable_cpu_data()[i] = caffe_cpu_asum(channels,
        diff_.cpu_data() + (i*channels));

    printf("CLL_CU: values of L1 norm are , %f \n", (float) dist_sq_.mutable_cpu_data()[i]);
    printf("CLL_CU: label : %d \n", static_cast<int>(bottom[2]->cpu_data()[i]));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs

      loss += Dtype(2) / margin * dist_sq_.cpu_data()[i] * dist_sq_.cpu_data()[i];
        printf(" CLL_CU: loss computed : %f\n", (float) loss);
    } else {  // dissimilar pairs
       printf("CLL_CU : the exponent of 1 is : %f \n",exp(Dtype(1)));
        printf("CLL_CU: value of L1 norm 2nd time is , %f \n", (float) dist_sq_.cpu_data()[i]);
        printf("CLL_CU: the value of margin is : %f \n", (float) margin);
      loss += Dtype(2) * margin * exp(-Dtype(2.77) / margin * dist_sq_.cpu_data()[i]);
        printf(" CLL_CU: loss computed : %f\n", (float) loss);
    }
  }

  loss = loss / static_cast<Dtype>(bottom[0]->num());
  printf("CLL_CU: value of loss : %f \n", loss);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

/* original backward_gpu method
template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = (*bottom)[0]->count();
      const int channels = (*bottom)[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>((*bottom)[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, alpha,
          (*bottom)[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          (*bottom)[i]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
    }
  }
}
*/


template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  printf("Running Backward GPU method !!!\n");
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = (*bottom)[0]->count();
      const int channels = (*bottom)[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      //margin = Dtype(1000);
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>((*bottom)[0]->num());

      int num = (*bottom)[i]->num();

    for (int j = 0; j < num; ++j) {
        Dtype* bout = (*bottom)[i]->mutable_cpu_diff();
        if (static_cast<int>((*bottom)[2]->cpu_data()[j])) {  // similar pairs
          for(int k = 0 ; k < channels ; k ++){
            Dtype gradient_sign = diff_.cpu_data()[(j*channels) + k] > 0 ? 1 : -1;
            bout[(j*channels) + k] += alpha * dist_sq_.mutable_cpu_data()[j] 
                                    * gradient_sign * 4 / margin;
          }
        } else {  // dissimilar pairs


          
          for(int k = 0 ; k < channels ; k ++){
            Dtype gradient_sign = diff_.cpu_data()[(j*channels) + k] > 0 ? 1 : -1;
            bout[(j*channels) + k] += alpha * Dtype(2) * -Dtype(2.77) 
                                    * exp(-Dtype(2.77) / margin * dist_sq_.mutable_cpu_data()[j] )
                                    * gradient_sign;
          }
        }
      }


      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_CLASS(ContrastiveLossLayer);

}  // namespace caffe
