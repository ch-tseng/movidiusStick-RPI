import numpy as np
import cv2
import io

class ssdMobilenets:
    def __init__(self, device, graphPath, Labels):
        # read in the graph file to memory buffer
        with open(graphPath, mode='rb') as f:
            graph_in_memory = f.read()
        graph = device.AllocateGraph(graph_in_memory)

        self.graph = graph
        self.labels = Labels

    def run(self, captured, confident):
        confident = confident * 100
        # The graph file that was created with the ncsdk compiler
        ssdMobilenets_graph = self.graph

        # read the image to run an inference on from the disk
        infer_image = captured

        # run a single inference on the image and overwrite the
        # boxes and labels
        img = self.__run_inference(infer_image, ssdMobilenets_graph, confident)
        img = img[:, :, ::-1]
        return img

    # Run an inference on the passed image
    # image_to_classify is the image on which an inference will be performed
    #    upon successful return this image will be overlayed with boxes
    #    and labels identifying the found objects within the image.
    # ssd_mobilenet_graph is the Graph object from the NCAPI which will
    #    be used to peform the inference.
    def __run_inference(self, image_to_classify, ssd_mobilenet_graph, confident):

        resized_image = self.__preprocess_image(image_to_classify)
        ssd_mobilenet_graph.LoadTensor(resized_image.astype(np.float16), None)
        output, userobj = ssd_mobilenet_graph.GetResult()
        num_valid_boxes = int(output[0])
        print('total num boxes: ' + str(num_valid_boxes))

        for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not np.isfinite(output[base_index]) or
                    not np.isfinite(output[base_index + 1]) or
                    not np.isfinite(output[base_index + 2]) or
                    not np.isfinite(output[base_index + 3]) or
                    not np.isfinite(output[base_index + 4]) or
                    not np.isfinite(output[base_index + 5]) or
                    not np.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
            x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[0]))
            y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[1]))
            x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[0]))
            y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[1]))

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            print('box at index: ' + str(box_index) + ' : ClassID: ' + self.labels[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')

            image_to_classify = self.__overlay_on_image(image_to_classify, output[base_index:base_index + 7], confident)

         # overlay boxes and labels on the original image to classify
        return image_to_classify

    def __overlay_on_image(self, display_image, object_info, confident):

        # the minimal score for a box to be shown
        min_score_percent = confident

        source_image_width = display_image.shape[1]
        source_image_height = display_image.shape[0]

        base_index = 0
        class_id = object_info[base_index + 1]
        percentage = int(object_info[base_index + 2] * 100)
        if (percentage <= min_score_percent):
            # ignore boxes less than the minimum score
            return

        label_text = self.labels[int(class_id)] + " (" + str(percentage) + "%)"
        box_left = int(object_info[base_index + 3] * source_image_width)
        box_top = int(object_info[base_index + 4] * source_image_height)
        box_right = int(object_info[base_index + 5] * source_image_width)
        box_bottom = int(object_info[base_index + 6] * source_image_height)

        box_color = (255, 128, 0)  # box color
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (125, 175, 75)
        label_text_color = (255, 255, 255)  # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                      label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_text_color, 1)

        return display_image

    def __preprocess_image(self, src):

        # scale the image
        NETWORK_WIDTH = 300
        NETWORK_HEIGHT = 300
        img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

        # adjust values to range between -1.0 and + 1.0
        img = img - 127.5
        img = img * 0.007843
        return img


class tinyYOLO2:
    def __init__(self, device, graphPath, Labels):
        # read in the graph file to memory buffer
        with open(graphPath, mode='rb') as f:
            graph_in_memory = f.read()
        graph = device.AllocateGraph(graph_in_memory)

        self.graph = graph
        self.labels = Labels

        # Tiny Yolo assumes input images are these dimensions.
        self.NETWORK_IMAGE_WIDTH = 448
        self.NETWORK_IMAGE_HEIGHT = 448

    def run(self, captured, confident=0.07):
        graph = self.graph
        # Load tensor and get result.  This executes the inference on the NCS
        input_image = captured
        display_image = input_image
        input_image = cv2.resize(input_image, (self.NETWORK_IMAGE_WIDTH, self.NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
        input_image = input_image.astype(np.float32)
        input_image = np.divide(input_image, 255.0)
        input_image = input_image[:, :, ::-1]  # convert to RGB

        graph.LoadTensor(input_image.astype(np.float16), 'user object')
        output, userobj = graph.GetResult()
        filtered_objs = self.__filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0], confident) # fc27 instead of fc12 for yolo_small
        img = self.__display_objects_in_gui(display_image, filtered_objs)
        #print(img.shape())
        return img


    def __filter_objects(self, inference_result, input_image_width, input_image_height, confident):

        # the raw number of floats returned from the inference (GetResult())
        num_inference_results = len(inference_result)

        # the 20 classes this network was trained on
        network_classifications = self.labels

        # only keep boxes with probabilities greater than this
        probability_threshold = confident

        num_classifications = len(network_classifications) # should be 20
        grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
        boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell

        # grid_size is 7 (grid is 7x7)
        # num classifications is 20
        # boxes per grid cell is 2
        all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

        # classification_probabilities  contains a probability for each classification for
        # each 64x64 pixel square of the grid.  The source image contains
        # 7x7 of these 64x64 pixel squares and there are 20 possible classifications
        classification_probabilities = \
            np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
        num_of_class_probs = len(classification_probabilities)

        # The probability scale factor for each box
        box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

        # get the boxes from the results and adjust to be pixel units
        all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
        all_boxes = self.__boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

        # adjust the probabilities with the scaling factor
        for box_index in range(boxes_per_grid_cell): # loop over boxes
            for class_index in range(num_classifications): # loop over classifications
                all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


        probability_threshold_mask = np.array(all_probabilities>=probability_threshold, dtype='bool')
        box_threshold_mask = np.nonzero(probability_threshold_mask)
        boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        probabilities_above_threshold = all_probabilities[probability_threshold_mask]

        # sort the boxes from highest probability to lowest and then
        # sort the probabilities and classifications to match
        argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
        boxes_above_threshold = boxes_above_threshold[argsort]
        classifications_for_boxes_above = classifications_for_boxes_above[argsort]
        probabilities_above_threshold = probabilities_above_threshold[argsort]


        # get mask for boxes that seem to be the same object
        duplicate_box_mask = self.__get_duplicate_box_mask(boxes_above_threshold)

        # update the boxes, probabilities and classifications removing duplicates.
        boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
        classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
        probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

        classes_boxes_and_probs = []
        for i in range(len(boxes_above_threshold)):
            classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

        return classes_boxes_and_probs

    # creates a mask to remove duplicate objects (boxes) and their related probabilities and classifi$
    # that should be considered the same object.  This is determined by how similar the boxes are
    # based on the intersection-over-union metric.
    # box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
    def __get_duplicate_box_mask(self, box_list):
        # The intersection-over-union threshold to use when determining duplicates.
        # objects/boxes found that are over this threshold will be
        # considered the same object
        max_iou = 0.35

        box_mask = np.ones(len(box_list))

        for i in range(len(box_list)):
            if box_mask[i] == 0: continue
            for j in range(i + 1, len(box_list)):
                if self.__get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                    box_mask[j] = 0.0

        filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
        return filter_iou_mask

    def __boxes_to_pixel_units(self, box_list, image_width, image_height, grid_size):

        # number of boxes per grid cell
        boxes_per_cell = 2

        # setup some offset values to map boxes to pixels
        # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
        box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

        # adjust the box center
        box_list[:,:,:,0] += box_offset
        box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
        box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

        # adjust the lengths and widths
        box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
        box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

        #scale the boxes to the image size in pixels
        box_list[:,:,:,0] *= image_width
        box_list[:,:,:,1] *= image_height
        box_list[:,:,:,2] *= image_width
        box_list[:,:,:,3] *= image_height

        return box_list

    def __get_intersection_over_union(self, box_1, box_2):

        # one diminsion of the intersecting box
        intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                             max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

        # the other dimension of the intersecting box
        intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                             max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

        if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
            # no intersection area
            intersection_area = 0
        else :
            # intersection area is product of intersection dimensions
            intersection_area =  intersection_dim_1*intersection_dim_2

        # calculate the union area which is the area of each box added
        # and then we need to subtract out the intersection area since
        # it is counted twice (by definition it is in each box)
        union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area;

        # now we can return the intersection over union
        iou = intersection_area / union_area
 
        return iou

    def __display_objects_in_gui(self, source_image, filtered_objects):
        # copy image so we can draw on it. Could just draw directly on source image if not concerned about that.
        display_image = source_image.copy()
        source_image_width = source_image.shape[1]
        source_image_height = source_image.shape[0]

        x_ratio = float(source_image_width) / self.NETWORK_IMAGE_WIDTH
        y_ratio = float(source_image_height) / self.NETWORK_IMAGE_HEIGHT

        # loop through each box and draw it on the image along with a classification label
        print('Found this many objects in the image: ' + str(len(filtered_objects)))
        for obj_index in range(len(filtered_objects)):
            center_x = int(filtered_objects[obj_index][1] * x_ratio)
            center_y = int(filtered_objects[obj_index][2] * y_ratio)
            half_width = int(filtered_objects[obj_index][3] * x_ratio)//2
            half_height = int(filtered_objects[obj_index][4] * y_ratio)//2

            # calculate box (left, top) and (right, bottom) coordinates
            box_left = max(center_x - half_width, 0)
            box_top = max(center_y - half_height, 0)
            box_right = min(center_x + half_width, source_image_width)
            box_bottom = min(center_y + half_height, source_image_height)

            print('box at index ' + str(obj_index) + ' is ' + filtered_objects[obj_index][0] + '( %.2f' % filtered_objects[obj_index][5] + ') left: ' + str(box_left) + ', top: ' + str(box_top) + ', right: ' + str(box_right) + ', bottom: ' + str(box_bottom))

            #draw the rectangle on the image.  This is hopefully around the object
            box_color = (0, 255, 0)  # green box
            box_thickness = 2
            cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

            # draw the classification label string just above and to the left of the rectangle
            label_background_color = (70, 120, 70) # greyish green background for text
            label_text_color = (255, 255, 255)   # white text
            cv2.rectangle(display_image,(box_left, box_top-20),(box_right,box_top), label_background_color, -1)
            cv2.putText(display_image,filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5], (box_left+5,box_top-7),cv2.FONT_HERSHEY_DUPLEX, 0.8, label_text_color, 1)


        display_image = display_image[...,::-1]
        return display_image
