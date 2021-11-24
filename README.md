# Official implementation for **ATDOC**

[**[CVPR-2021] Domain Adaptation with Auxiliary Target Domain-Oriented Classifier**](https://arxiv.org/pdf/2007.04171.pdf)

[Update @ Nov 23 2021] 

1. **[For Office, please change the *max-epoch* to 100; for VISDA-C, change the *max-epoch* to 1 and change the *net* to resnet101]**
2. **Add the code associated with SSDA, change the *max-epoch* to 20 for DomainNet-126**
3. **Thank @lyxok1 for pointing out the typo in Eq.(6), we have corrected it in the new verison of this paper.**



Below is the demo for **ATDOC** on a UDA task of Office-Home [*max_epoch* to 50]:


1. installing packages

   `python == 3.6.8`
   `pytorch ==1.1.0`
   `torchvision == 0.3.0`
   `numpy, scipy, sklearn, PIL, argparse, tqdm`
   
2. download the Office-Home dataset

   `mkdir dataset`

   `cd dataset`

   `pip install gdown`

   `gdown https://drive.google.com/u/0/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg&export=download`

   `unzip OfficeHomeDataset_10072016.zip`

   `mv ./OfficeHomeDataset_10072016/Real\ World ./OfficeHomeDataset_10072016/RealWorld`

   `cd ../`

3. run the main file with '**Source-model-only**'

   `python demo_uda.py --pl none --dset office-home --max_epoch 50 --s 0 --t 1 --gpu_id 0 --method srconly --output logs/uda/run1/`

4. run the main file with '**ATDOC-NC**'

   `python demo_uda.py --pl atdoc_nc --tar_par 0.1 --dset office-home --max_epoch 50 --s 0 --t 1 --gpu_id 0 --method srconly --output logs/uda/run1/`

5. run the main file with '**ATDOC-NA**'

   `python demo_uda.py --pl atdoc_na --tar_par 0.2 --dset office-home --max_epoch 50 --s 0 --t 1 --gpu_id 0 --method srconly --output logs/uda/run1/`

6. run the main file with '**ATDOC-NA**' combined with '**CDAN+E**'

   `python demo_uda.py --pl atdoc_na --tar_par 0.2 --dset office-home --max_epoch 50 --s 0 --t 1 --gpu_id 0 --method CDANE --output logs/uda/run1/`

7. run the main file with '**ATDOC-NA**' combined with '**MixMatch**'

   `python demo_mixmatch.py --pl none --dset office-home  --max_epoch 50 --s 0 --t 1 --gpu_id 0 --output logs/uda/run1/`

8. run the main file with '**ATDOC-NA**' combined with '**MixMatch**'

   `python demo_mixmatch.py --pl atdoc_na --dset office-home  --max_epoch 50 --s 0 --t 1 --gpu_id 0 --output logs/uda/run1/`




### Citation

If you find this code useful for your research, please cite our paper

> @inproceedings{liang2021domain,  
>  &nbsp; &nbsp;  title={Domain Adaptation with Auxiliary Target Domain-Oriented Classifier},  
>  &nbsp; &nbsp;  author={Liang, Jian and Hu, Dapeng and Feng, Jiashi},  
>  &nbsp; &nbsp;  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
>  &nbsp; &nbsp;  year={2021}  
> }
> 
### Contact

- [liangjian92@gmail.com](mailto:liangjian92@gmail.com)
- [dapeng.hu@u.nus.edu](mailto:dapeng.hu@u.nus.edu)
- [elefjia@nus.edu.sg](mailto:elefjia@nus.edu.sg)