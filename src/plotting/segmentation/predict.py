import multiprocessing
import os

import importlib
import numpy as np
import pandas as pd
import cli_pred as cp
importlib.reload(cp)
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import segmentation, feature

class PQCalculator:
    def __init__(self, ids):
        self.ids = ids
        self.length = len(ids)
        self.labels = list(range(0, 256 ** 2, 1))


    def calc_sem_iou(self, gt, mask, num_classes=None, ignore_index=None):
        ious = {}
        dices = {}
        gt = np.array(gt)
        mask = np.array(mask)

        classes = np.unique(gt)
        #print(f"Unique classes in GT: {classes}")

        if ignore_index is not None:
            classes = classes[classes != ignore_index]

        for c in classes:
            gt_c = (gt == c)
            pred_c = (mask == c)

            intersection = np.logical_and(gt_c, pred_c).sum()
            union = np.logical_or(gt_c, pred_c).sum()

            iou = intersection / union if union > 0 else 0
            dice = (2 * intersection) / (gt_c.sum() + pred_c.sum()) if (gt_c.sum() + pred_c.sum()) > 0 else 0

            ious[c] = iou
            dices[c] = dice

        if len(ious) > 0:
            mean_iou = np.mean(list(ious.values()))
        else:
            mean_iou = 0.0

        if len(dices) > 0:
            mean_dice = np.mean(list(dices.values()))
        else:            
            mean_dice = 0.0

        binary_gt = (gt != 0)
        binary_mask = (mask != 0)
        intersection = np.logical_and(binary_gt, binary_mask).sum()
        union = np.logical_or(binary_gt, binary_mask).sum()

        binary_iou = intersection / union if union > 0 else 0
        binary_dice = (2 * intersection) / (binary_gt.sum() + binary_mask.sum()) if (binary_gt.sum() + binary_mask.sum()) > 0 else 0

        #print(f"Class IoUs: {ious.items()}, Mean IoU: {mean_iou}")
        #print(f"Class Dices: {dices.items()}, Mean Dice: {mean_dice}")
        #print(f"Binary IoU: {binary_iou}, Binary Dice: {binary_dice}")

        return ious, mean_iou, binary_iou, dices, mean_dice, binary_dice



    def calc_PQ(self, ids):
        with multiprocessing.Manager() as manager:
            qL = manager.list()
            mL_ins = manager.list()
            sem_seg = []
            inst_seg = []

            CD = os.path.dirname(os.path.abspath("temp.ipynb"))
            patch_names = pd.read_csv(os.path.join(os.path.dirname(CD), "CRC-HnE-Segmentation-Training", "histology_segmentation_training", "data", "patches", "patch_info.csv"))

            #model = cp.get_pytorch_model("../CRC-HnE-Segmentation-Training/mlruns/unet_fromC32_Aug_backgound0_5/best_Unet_lr-0.003_wd-1e-05_dropout-0.015_epoch-250_batchS-64-v3.ckpt", False, "U-Net")
            model = cp.get_pytorch_model("U_NET.ckpt", False, "U-Net")
            #model = cp.get_pytorch_model("../CRC-HnE-Segmentation-Training/mlruns/unext_fromC32_Aug_backgroud1/best_UneXt_lr-0.003_wd-1e-05_dropout-0.015_epoch-250_batchS-32.ckpt", False, "U-NeXt")
            #model = cp.get_pytorch_model("../CRC-HnE-Segmentation-Training/mlruns/swinunetr_fromC32_Aug_background0_5/best_swinUNETR_lr-0.0005_wd-1e-05_dropout-0.01_epoch-250_batchS-16-v1.ckpt", False, "swin-UNETR")
            #model = cp.get_pytorch_model("../CRC-HnE-Segmentation-Training/mlruns/unet_deeper/best_Unet_lr-0.0001_wd-1e-05_dropout-0.02_epoch-300_batchS-32-v1.ckpt", False, "U-Net")

        

            for idx in range(len(ids)):
                sem_seg, inst_seg = self.process_pipeline(os.path.join(os.path.dirname(CD), "CRC-HnE-Segmentation-Training", "histology_segmentation_training", "data", "OME-TIFFs", patch_names.iloc[ids[idx], 0] + ".ome.tif"), patch_names.iloc[ids[idx], 0], 0, qL, mL_ins, sem_seg, inst_seg, model)
            
        self.segs = np.array(sem_seg)
        self.inst_segs = np.array(inst_seg)
        np.save("metrics_sem_segs", self.segs)
        np.save("metrics_inst_segs", self.inst_segs)

    def load_image(self, path, mask: bool):
        data = tiff.imread(path)
        label = data[3, :, :]
        image = data[:3, :, :]
        if mask:
            return label
        return image, label

    def read_data_to_predict(self, path):
        data = tiff.imread(path)
        label = data[3, :, :]
        image = data[:3, :, :]
        return image, label

    def process_pipeline(self, iD, val, cl, qL, ml_ins, sem_seg, inst_seg, model):
        image, label = self.read_data_to_predict(iD)
        result = cp.predict(image, model)
        result = np.argmax(result.detach().numpy(), axis=1)

        sem_metrics = self.calc_sem_iou(label, result)

        inst_metrics = []
        for class_label in range(1, 7):
            inst_mask = self.watershed_instances(result[0], class_label)
            inst_gt = self.watershed_instances(label, class_label)
            rq, sq, pq = self.intersection_two_instances(inst_gt, inst_mask)
            inst_metrics.append({"rq": rq, "sq": sq, "pq": pq})
          
        sem_seg.append(sem_metrics)
        inst_seg.append(inst_metrics)
        return sem_seg, inst_seg


    def watershed_instances(self, mask, class_label):

        nuclei_mask = (mask == class_label)
        
        if np.sum(nuclei_mask) == 0:
            return np.zeros_like(mask, dtype=int)

        distance = ndi.distance_transform_edt(nuclei_mask)

        coords = feature.peak_local_max(
            distance,
            labels=nuclei_mask,
            min_distance=3,  
        )

        markers = np.zeros_like(mask, dtype=int)
        if len(coords) > 0:
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

        inst_mask = segmentation.watershed(-distance, markers, mask=nuclei_mask)

        return inst_mask
    

    def compute_RQ_SQ_PQ(self, tp, fp, fn, tp_ious):

        if tp == 0 and fp == 0 and fn == 0:
            return None, None, None
        
        if tp + 0.5 * fp + 0.5 * fn > 0:
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        else:
            rq = 0.0

        if tp > 0:
            sq = tp_ious / tp
        else:
            sq = 0.0

        pq = rq * sq

        return rq, sq, pq

    def intersection_two_instances(self, ground_truth, mask):
        tp = 0
        fp = 0
        tp_ious = 0
        
        gt_instances = list(np.unique(ground_truth))[1:]
        pred_instances = list(np.unique(mask))[1:]

        matched_preds = set()  
        fns = set(gt_instances)  

        for gt_val in gt_instances:
            gt_mask = ground_truth == gt_val
            best_iou = 0
            best_pred = None

            # find the predicted instance with highest IoU
            for pred_val in pred_instances:
                if pred_val in matched_preds:
                    continue  # already matched predictions

                pred_mask = mask == pred_val
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = intersection / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_val

            # evaluate the best match
            if best_iou >= 0.5:  
                tp += 1
                tp_ious += best_iou
                matched_preds.add(best_pred)
                fns.discard(gt_val)  # remove matched ground truth val
            else:
                # no good match
                continue

        # unmatched predictions as FP
        fp = len(set(pred_instances) - matched_preds)

        rq, sq, pq = self.compute_RQ_SQ_PQ(tp, fp, len(fns), tp_ious)

        #print(f"RQ: {rq}, SQ: {sq}, PQ: {pq}")

        return rq, sq, pq
    


if __name__ == '__main__':
    
    test_id = [166, 2197, 1519, 1654, 2986, 4891, 3021, 1536, 653, 279, 1405, 605, 4336, 3095, 1209, 350, 19, 4485, 247,
               4240, 1589, 2481, 2759, 3836, 2629, 1338, 3935, 3932, 356, 3429, 3277, 4502, 3624, 1652, 463, 2311, 4906,
               4418, 4679, 677, 1557, 1503, 2462, 535, 755, 254, 2307, 309, 4133, 3250, 177, 198, 4946, 4923, 413, 4388,
               3339, 4005, 1592, 1945, 1024, 1051, 2110, 1034, 3880, 3599, 4184, 4596, 2510, 3595, 1044, 1164, 1569,
               4611, 3800, 596, 1526, 501, 4209, 1210, 2194, 4023, 1534, 4063, 149, 387, 4214, 615, 549, 3153, 4258,
               4571, 765, 1595, 3018, 507, 2576, 3754, 4511, 4130, 228, 4873, 3892, 1554, 4512, 1187, 4567, 3185, 2395,
               1504, 2984, 1371, 4445, 4814, 486, 4882, 2297, 2053, 3906, 2702, 1566, 2527, 4003, 436, 4840, 2865, 2575,
               511, 3994, 3834, 157, 805, 4236, 490, 3720, 1803, 3570, 4411, 2910, 2732, 679, 4947, 2521, 4379, 4352,
               594, 4774, 4903, 1046, 2902, 4837, 4575, 4256, 188, 3845, 945, 862, 4557, 2847, 4597, 4036, 787, 93,
               1322, 3138, 1095, 305, 643, 4561, 887, 1255, 3370, 2534, 4450, 4062, 274, 1480, 3752, 2881, 2252, 2177,
               2346, 1618, 214, 3411, 2211, 3706, 565, 4956, 1212, 433, 1590, 4576, 1378, 3313, 2754, 2406, 1215, 208,
               1937, 4304, 371, 1891, 2144, 2004, 1032, 4398, 841, 211, 633, 122, 3887, 1808, 4190, 1600, 611, 1407,
               298, 705, 3768, 620, 3779, 1101, 1451, 1157, 3247, 3864, 445, 2100, 2767, 2331, 627, 1497, 1609, 1815,
               3883, 4665, 1434, 2220, 731, 3221, 3328, 3188, 1820, 2416, 1736, 3084, 1513, 111, 152, 4638, 4783, 4461,
               1412, 3983, 670, 3567, 4595, 494, 4654, 2995, 2215, 3832, 1543, 3428, 4157, 230, 2101, 3699, 4422, 3997,
               937, 461, 1437, 802, 527, 1351, 1479, 438, 1221, 73, 3733, 576, 465, 354, 248, 2464, 4257, 888, 450,
               2907, 144, 330, 1258, 1650, 517, 4098, 3543, 415, 561, 2794, 2921, 1469, 1113, 2990, 4670, 4542, 3494,
               1545, 3001, 915, 183, 1260, 2648, 3580, 3784, 3291, 790, 106, 2063, 4895, 4293, 4883, 497, 1818, 1084,
               3091, 2699, 4351, 256, 2638, 135, 1739, 2426, 3112, 191, 2748, 472, 506, 4080, 1666, 422, 1238, 1075,
               3896, 1231, 4085, 3227, 1323, 3634, 4217, 2615, 964, 2309, 457, 69, 828, 2369, 3203, 1907, 4976, 4375,
               680, 2445, 95, 3588, 4197, 2432, 1878, 1052, 2233, 2857, 2879, 1146, 1477, 2404, 952, 2083, 4842, 1362,
               3214, 2594, 2389, 332, 2059, 175, 1204, 879, 1242, 2672, 3919, 2592, 2103, 4508, 45, 3601, 3365, 353,
               4483, 2647, 1174, 4393, 3179, 4830, 70, 2337, 4354, 996, 3825, 1756, 3305, 794, 3026, 1919, 2471, 2303,
               1832, 4687, 2900, 1912, 4396, 1729, 911, 1281, 2765, 978, 4074, 711, 3757, 3539, 1144, 1932, 4409, 3270,
               3473, 831, 1718, 4490, 1293, 4683, 1620, 2409, 1786, 1483, 1482, 4634, 1397, 2275, 4943, 586, 4693, 4122,
               3749, 734, 1578, 3610, 4104, 859, 4608, 1588, 2052, 3080, 3703, 1391, 3349, 1615, 681, 252, 1117, 1426,
               4909, 4457, 1029, 3694, 742, 1773, 2287, 3050, 644, 4235, 1010, 179, 4326, 2915, 1957, 1964, 3742, 2837,
               1747, 2142, 360, 4942, 553, 2656, 4920, 2492, 2339, 3622, 3782, 4385, 4417, 1807, 733, 468, 2991, 2588,
               3905, 801, 1108, 3366, 170, 3801, 810, 1272, 4666, 518, 534, 3238, 725, 3482, 2088, 1752, 3044, 912,
               4439, 3591, 2804, 881, 3380, 2741, 3778, 3604, 4898, 4586, 4940, 2581, 259, 2498, 812, 2547, 293, 2812,
               3666, 4311, 2977, 2133, 1444, 564, 1813, 690, 4835, 1538, 2569, 4172, 2925, 4570, 2858, 2450, 2473, 4572,
               3447, 3811, 1873, 2304, 4335, 283, 2093, 1954, 4162, 4444, 2683, 4849, 530, 2119, 2728, 3526, 2932, 3550,
               2825, 192, 1417, 1175, 2655, 4710, 2072, 4625, 47, 2602, 2770, 4633, 2600, 2627, 1782, 1149, 4646, 1550,
               893, 1721, 4024, 278, 2372, 4753, 4479, 907, 3691, 2437, 2078, 691, 3586, 718, 2238, 3639, 3437, 2520,
               2456, 3659, 2603, 3554, 4701, 4230, 4420, 2284, 4359, 3646, 2111, 1539, 1564, 3268, 1870, 14, 4901, 4462,
               4205, 3770, 328, 3484, 625, 3934, 648, 1295, 1583, 3682, 3265, 544, 4044, 2153, 1128, 1041, 4763, 505,
               1769, 1616, 2892, 1886, 1074, 1738, 270, 4467, 3128, 798, 2698, 3263, 3043, 4119, 2092, 3886, 1532, 977,
               4281, 4017, 3353, 1614, 308, 1675, 2845, 4222, 4655, 3264, 1335, 896, 4046, 4255, 3988, 1361, 2317, 2118,
               4922, 1360, 3231, 598, 2277, 2519, 1433, 3331, 4207, 4711, 3984, 227, 184, 1849, 2771, 1751, 3625, 4629,
               1610, 4221, 3244, 2874, 1744, 1436, 4793, 3759, 4246, 3638, 1840, 1833, 3002, 68, 1746, 1189, 33, 3198,
               3918, 4515, 2518, 4977, 1438, 2042, 1926, 2002, 624, 3338, 2373, 378, 3296, 1961, 2587, 376, 708, 3740,
               4049, 4114, 287, 3269, 4544, 1413, 4347, 2402]  

    
    #test_id = [605]
    
    pqc = PQCalculator(test_id)
    pqc.calc_PQ(test_id)