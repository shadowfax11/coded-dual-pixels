from .model_dddnet import Mydeblur
from .model_dddnet import YRStereonet_3D

class DDDNet(nn.Module):

    def __init__(self, in_channels=6, dim, out_dim):
        super(BasicConv, self).__init__()
        self.Estd_stereo = model_test.YRStereonet_3D()
        self.Esti_stereod = model_test.Mydeblur()
    
    def forward(self,x):
        image_left = x[:,::2,:,:]
        image_right = x[:,1::2,:,:]
        est_blurdispt = Estd_stereo(image_left,image_right)
        deblur_imaget, est_mdispt = Esti_stereod(image_left, image_right, est_blurdispt) 

        return est_blurdisp,est_mdisp,deblur_image
        

