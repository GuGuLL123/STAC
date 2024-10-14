
class Config:
    def __init__(self,task):
        if task == "synapse":
            self.base_dir = '/data/ylgu/Medical/Semi_medical_data/Synapse_semi/RawData'
            self.save_dir = '/data/ylgu/Medical/Semi_medical_data/Synapse_semi/ProcessedData'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
        else: # amos
            self.base_dir = '/data/ylgu/Medical/Semi_medical_data/amos22'
            self.save_dir = '/data/ylgu/Medical/Semi_medical_data/amos22_processed'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50