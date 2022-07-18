def sample_desc_to_xml_path(df, sample_desc,img_key):
    xml_path = df[img_key][df.sample_desc == sample_desc].values
    return xml_path
def get_gt_from_tabular_sample(tabular_sample_dict,gt_key):
    gt = tabular_sample_dict[gt_key]
    tabular_sample_dict.pop(gt_key)
    return tabular_sample_dict,gt

class ImagingTabularProcessor:
    def __init__(self, data, label,img_key,image_processor, tabular_processor):
        self.image_processor = image_processor
        self.tabular_processor = tabular_processor
        self.data = data
        self.label = label
        self.img_key = img_key
    def __call__(self, sample_desc):
        img_path = sample_desc_to_xml_path(self.data, sample_desc,self.img_key)
        tabular_sample_dict = self.tabular_processor(sample_desc)
        image_dict_list = self.image_processor(img_path[0][0])
        tabular_sample_dict,gt = get_gt_from_tabular_sample(tabular_sample_dict.copy(), self.label)
        img_sample_dict = image_dict_list
        sample = tabular_sample_dict
        sample['image'] = img_sample_dict
        sample['gt'] = gt
        return sample