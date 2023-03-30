import numpy as np

from typing import Any
from tracking.assignment import Assignment

class Association:

    def __init__(self) -> None:
        self.assignment = Assignment()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    @staticmethod
    def iou_batch(bb_test, bb_gt):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2,id]
        Given [N, 5] [M, 5] -> []
        """

        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        test_width = bb_test[..., 2] - bb_test[..., 0]
        test_height = bb_test[..., 3] - bb_test[..., 1]
        gt_width = bb_gt[..., 2] - bb_gt[..., 0]
        gt_height = bb_gt[..., 3] - bb_gt[..., 1]
        o = wh / (test_width * test_height                                      
        + gt_width * gt_height - wh)                                              
        return o

    @staticmethod
    def batch_iou(a, b, epsilon=1e-5):
        """ Given two arrays `a` and `b` where each row contains a bounding
            box defined as a list of four numbers:
                [x1,y1,x2,y2]
            where:
                x1,y1 represent the upper left corner
                x2,y2 represent the lower right corner
            It returns the Intersect of Union scores for each corresponding
            pair of boxes.

        Args:
            a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
            b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
            epsilon:    (float) Small value to prevent division by zero

        Returns:
            (numpy array) The Intersect of Union scores for each pair of bounding
            boxes.
        """
        # COORDINATES OF THE INTERSECTION BOXES
        x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
        y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
        x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
        y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

        # AREAS OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)

        # handle case where there is NO overlap
        width[width < 0] = 0
        height[height < 0] = 0

        area_overlap = width * height

        # COMBINED AREAS
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined + epsilon)
        return iou

    @staticmethod
    def batch_iou_ext_sep(bb_det, bb_trk, ext_w, ext_h):
        """
        Computes extended IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
        with separate extension coefficient

        @param bb_det: detection bounding box => [N, 4]
        @param bb_trk: tracking bounding box => [M, 4]
        @param ext_w: extension coefficient for width
        @param ext_h: extension coefficient for height


        @return: extended IOU => [N, M]
        """
        trk_w = bb_trk[..., 2] - bb_trk[..., 0]
        trk_h = bb_trk[..., 3] - bb_trk[..., 1]
        xx1 = np.maximum(bb_det[..., 0], bb_trk[..., 0] - trk_w * ext_w/2)
        xx2 = np.minimum(bb_det[..., 2], bb_trk[..., 2] + trk_w * ext_w/2)
        w = np.maximum(0., xx2 - xx1)
        yy1 = np.maximum(bb_det[..., 1], bb_trk[..., 1] - trk_h * ext_h/2)
        yy2 = np.minimum(bb_det[..., 3], bb_trk[..., 3] + trk_h * ext_h/2)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        area_det = (bb_det[..., 2] - bb_det[..., 0]) * (bb_det[..., 3] - bb_det[..., 1])
        area_trk = (bb_trk[..., 2] - bb_trk[..., 0]) * (bb_trk[..., 3] - bb_trk[..., 1])
        o = wh / (area_det + area_trk - wh)
        return o

    @staticmethod
    def batch_outside(trk, img_s):
        """
        Computes how many percent of trk is placed outside of img_s

        @param trk: tracking bounding box => [N, 4]
        @param img_s: image size => [2]

        @return: outside percent => [N]
        """
        out_x = np.zeros(len(trk))
        out_y = np.zeros(len(trk))
        
        mask_0 = trk[..., 0] < 0
        trk[mask_0, 0] = -trk[mask_0, 0]

        mask_1 = trk[..., 2] > img_s[0]
        out_x[mask_1] = trk[mask_1, 2] - img_s[0]

        mask_2 = trk[..., 1] < 0
        out_y[mask_2] = -trk[mask_2, 1]

        mask_3 = trk[..., 1] < 0
        out_y[mask_3] = -trk[mask_3, 1]

        mask_4 = trk[..., 3] > img_s[1]
        out_y[mask_4] = trk[mask_4, 3] - img_s[1]
        
        out_a = out_x * (trk[..., 3] - trk[..., 1]) + out_y * (trk[..., 2] - trk[..., 0])
        area = (trk[..., 3] - trk[..., 1]) * (trk[..., 2] - trk[..., 0])
        return out_a / area

    @staticmethod
    def ios_batch(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h

        test_width = bb_test[..., 2] - bb_test[..., 0]
        test_height = bb_test[..., 3] - bb_test[..., 1]

        o = wh / (test_width * test_height)

        return o

    def associate_detections_to_trackers(self, 
                                         detections,
                                         trackers,
                                         iou_threshold = 0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trackers)==0):
            # if no trackers, return empty all
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

        iou_matrix = Association.iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self.assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]]<iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def associate_detections_to_trackers_OH(self, mot_tracker, detections, trackers, groundtruths, average_area, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 5 lists of matches, unmatched_detections, unmatched_trackers, occluded_trackers and unmatched ground truths
        """
        if len(trackers) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 1), dtype=int), np.empty((0, 1), dtype=int), np.empty((0, 1), dtype=int)

        # assign only according to iou
        # calculate intersection over union cost
        iou_matrix = Association.iou_batch(detections, trackers)
        ios_matrix = Association.ios_batch(trackers)
        np.fill_diagonal(ios_matrix, 0)

        matched_indices = self.assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices)
        matched_indices = np.transpose(matched_indices)     # first column: detection indexes, second column: object indexes

        unmatched_detections = []
        det_ind = np.arange(len(detections))
        det_ind = det_ind[np.isin(det_ind, matched_indices[:, 0], invert=True)]
        # for d, det in enumerate(detections):
        #     if d not in matched_indices[:, 0]:
        #         unmatched_detections.append(d)

        unmatched_trackers = []
        trk_ind = np.arange(len(trackers))
        trk_ind = trk_ind[np.isin(trk_ind, matched_indices[:, 1], invert=True)]
        # for t, trk in enumerate(trackers):
        #     if t not in matched_indices[:, 1]:
        #         unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                mot_tracker.trackers[m[1]].time_since_observed = 0

        unmatched_detections = np.array(unmatched_detections)
        unmatched_trackers = np.array(unmatched_trackers)
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

            # try to match extended unmatched tracks to unmatched detections
            if len(unmatched_detections) > 0 and len(unmatched_trackers) > 0 and len(mot_tracker.unmatched_before) > 0:
                unm_dets = []
                for ud in unmatched_detections:
                    unm_dets.append(detections[ud])
                iou_matrix = Association.iou_batch(unm_dets, mot_tracker.unmatched_before)
                matched_indices = self.assignment(-iou_matrix)
                matched_indices = np.asarray(matched_indices)
                matched_indices = np.transpose(matched_indices)

                iou_matrix_ext = np.zeros((len(unmatched_trackers), len(unmatched_detections)), dtype=np.float32)
                for ud in range(len(unmatched_detections)):
                    for ut in range(len(unmatched_trackers)):
                        iou_matrix_ext[ut, ud] = Association.batch_iou_ext_sep(detections[unmatched_detections[ud]],
                                                                            trackers[unmatched_trackers[ut]],
                                                                            np.minimum(1.2, (mot_tracker.trackers[
                                                                                                unmatched_trackers[
                                                                                                    ut]].time_since_observed + 1) * 0.3),
                                                                            np.minimum(0.5, (mot_tracker.trackers[
                                                                                                unmatched_trackers[
                                                                                                    ut]].time_since_observed + 1) * 0.1))
                matched_indices_ext = self.assignment(-iou_matrix_ext)
                matched_indices_ext = np.asarray(matched_indices_ext)
                matched_indices_ext = np.transpose(matched_indices_ext)

                # filter out matched with low IOU and low area
                matches_ext_com = []
                del_ind = np.empty((0, 3), dtype=int)
                for m in matched_indices_ext:
                    ind = matched_indices[np.where(matched_indices[:, 0] == m[1]), 1]
                    if ind.size:
                        if (iou_matrix_ext[m[0], m[1]] >= iou_threshold) and (iou_matrix[m[1], ind] >= iou_threshold):
                            matches_ext_com.append(np.concatenate((m.reshape(1, 2), ind), axis=1))
                            # remove matched detections from unmatched arrays
                            del_ind = np.concatenate((del_ind, np.array([m[0], m[1], ind.item(0)]).reshape(1, 3)))

                to_del_und = []
                to_del_unt = []
                to_del_undb = []
                matches_ext = np.empty((0, 2), dtype=int)
                if len(matches_ext_com) > 0:
                    matches_ext_com = np.concatenate(matches_ext_com, axis=0)
                    for m in matches_ext_com:
                        new = np.array([[unmatched_detections[m[1]], unmatched_trackers[m[0]]]])
                        matches_ext = np.append(matches_ext, new, axis=0)
                        to_del_unt.append(m[0])
                        to_del_und.append(m[1])
                        to_del_undb.append(m[2])
                matches = np.concatenate((matches, matches_ext))
                if len(to_del_unt) > 0:
                    to_del_unt = np.array(to_del_unt)
                    to_del_unt = np.sort(to_del_unt)
                    for i in reversed(to_del_unt):
                        unmatched_trackers = np.delete(unmatched_trackers, i, 0)
                if len(to_del_und) > 0:
                    to_del_und = np.array(to_del_und)
                    to_del_und = np.sort(to_del_und)
                    for i in reversed(to_del_und):
                        unmatched_detections = np.delete(unmatched_detections, i, 0)
                if len(to_del_undb) > 0:
                    to_del_undb = np.array(to_del_undb)
                    to_del_undb = np.sort(to_del_undb)
                    for i in reversed(to_del_undb):
                        mot_tracker.unmatched_before.pop(i)

        occluded_trackers = []
        if mot_tracker.frame_count > mot_tracker.min_hits:
            trks_occlusion = np.amax(ios_matrix, axis=0)
            unm_trks = unmatched_trackers
            unmatched_trackers = []
            for ut in unm_trks:
                ut_area = (trackers[ut, 3] - trackers[ut, 1])*(trackers[ut, 2] - trackers[ut, 0])
                mot_tracker.trackers[ut].time_since_observed += 1
                mot_tracker.trackers[ut].confidence = min(1, mot_tracker.trackers[ut].age/(mot_tracker.trackers[ut].time_since_observed*10)*(ut_area/average_area))
                if trks_occlusion[ut] > 0.3 and mot_tracker.trackers[ut].confidence > mot_tracker.conf_trgt:
                # if trks_occlusion[ut] > 0.3 and ut_area > 0.7 * average_area and mot_tracker.trackers[ut].age > 5:
                    occluded_trackers.append(ut)
                elif mot_tracker.trackers[ut].confidence > mot_tracker.conf_objt:
                # elif mot_tracker.trackers[ut].age > (mot_tracker.trackers[ut].time_since_observed * 10 + 10) and mot_tracker.trackers[ut].time_since_observed < 5:
                    occluded_trackers.append(ut)
                else:
                    unmatched_trackers.append(ut)

        # find unmatched ground truths
        unmatched_groundtruths = []
        # if visualization.DisplayState.display_gt_diff:
        #     found_trackers = trackers
        #     for i in reversed(np.sort(unmatched_trackers)):
        #         found_trackers = np.delete(found_trackers, i, 0)
        #     iou_matrix_1 = Association.batch_iou(groundtruths, found_trackers)
        #     # first column: ground truth indexes, second column: object indexes
        #     matched_indices_1 = self.assignment(-iou_matrix_1)
        #     matched_indices_1 = np.asarray(matched_indices_1)
        #     matched_indices_1 = np.transpose(matched_indices_1)

        #     for g, gt in enumerate(groundtruths):
        #         if g not in matched_indices_1[:, 0]:
        #             unmatched_groundtruths.append(g)

        return matches, unmatched_detections, unmatched_trackers, np.array(occluded_trackers), np.array(unmatched_groundtruths)