class graphs:
  def __init__(self):
    self.accuracy     = []
    self.loss         = []
    self.reg_loss     = []

    # NC1
    self.Sw_invSb     = []

    # NC2
    self.norm_M_CoV   = []
    self.norm_W_CoV   = []
    self.cos_M        = []
    self.cos_W        = []

    # NC3
    self.W_M_dist     = []
    
    # NC4
    self.NCC_mismatch = []

    # Decomposition
    self.MSE_wd_features = []
    self.LNC1 = []
    self.LNC23 = []
    self.Lperp = []

class specs:
    def __init__(self):
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.criterion_summed = None
        self.epoch_list = None
        self.loss_name = None
        self.num_classes = None
        self.classifier = None
        self.weight_decay = None
        self.input_channels = None