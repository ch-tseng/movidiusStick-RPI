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


    def run(self, captured):
        # The graph file that was created with the ncsdk compiler
        ssdMobilenets_graph = self.graph

        # read the image to run an inference on from the disk
        infer_image = captured

        # run a single inference on the image and overwrite the
        # boxes and labels
        return self.__run_inference(infer_image, ssdMobilenets_graph)


    # Run an inference on the passed image
    # image_to_classify is the image on which an inference will be performed
    #    upon successful return this image will be overlayed with boxes
    #    and labels identifying the found objects within the image.
    # ssd_mobilenet_graph is the Graph object from the NCAPI which will
    #    be used to peform the inference.
    def __run_inference(self, image_to_classify, ssd_mobilenet_graph):

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

            # overlay boxes and labels on the original image to classify
            return self.__overlay_on_image(image_to_classify, output[base_index:base_index + 7])

    def __overlay_on_image(self, display_image, object_info):

        # the minimal score for a box to be shown
        min_score_percent = 20

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
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

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


