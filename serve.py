from __future__ import absolute_import, print_function
import click
import base64

import tempfile
import uuid

from tomaat.core.service import TOMAATService
import time
import tensorflow as tf
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from train import NetFactory


DESCRIPTION = """
    Prediction endpoint serving:
    Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional 
    Neural Networks." arXiv preprint arXiv:1710.04043 (2017).
"""

@click.group()
def cli():
    pass


def dummy_data_pipeline(data):
    return data


class Brats17(TOMAATService):
    # This class allows the brats17 model to be shared on the network.

    widgets = \
        [  # THIS defines the input interface of this service
            {'type': 'volume', 'destination': 'Flair'},  # FLAIR volume
            {'type': 'volume', 'destination': 'T1'},  # T1
            {'type': 'volume', 'destination': 'T1ce'},  # T1ce
            {'type': 'volume', 'destination': 'T2'},  # T2
        ]

    def __init__(self, config_file, **kwargs):

        super(Brats17, self).__init__(**kwargs)

        # 1, load configure file
        self.config = parse_config(config_file)
        self.config_data = self.config['data']
        self.config_net1 = self.config.get('network1', None)
        self.config_net2 = self.config.get('network2', None)
        self.config_net3 = self.config.get('network3', None)
        self.config_test = self.config['testing']
        self.batch_size = 5

        # 2.1, network for whole tumor
        if (self.config_net1):
            net_type1 = self.config_net1['net_type']
            net_name1 = self.config_net1['net_name']
            self.data_shape1 = self.config_net1['data_shape']
            self.label_shape1 = self.config_net1['label_shape']
            self.class_num1 = self.config_net1['class_num']

            # construct graph for 1st network
            full_data_shape1 = [self.batch_size] + self.data_shape1
            self.x1 = tf.placeholder(tf.float32, shape=full_data_shape1)
            net_class1 = NetFactory.create(net_type1)
            self.net1 = net_class1(num_classes=self.class_num1, w_regularizer=None,
                              b_regularizer=None, name=net_name1)
            self.net1.set_params(self.config_net1)
            predicty1 = self.net1(self.x1, is_training=True)
            self.proby1 = tf.nn.softmax(predicty1)
        else:
            config_net1ax = self.config['network1ax']
            config_net1sg = self.config['network1sg']
            config_net1cr = self.config['network1cr']

            # construct graph for 1st network axial
            net_type1ax = config_net1ax['net_type']
            net_name1ax = config_net1ax['net_name']
            self.data_shape1ax = config_net1ax['data_shape']
            self.label_shape1ax = config_net1ax['label_shape']
            self.class_num1ax = config_net1ax['class_num']

            full_data_shape1ax = [self.batch_size] + self.data_shape1ax
            self.x1ax = tf.placeholder(tf.float32, shape=full_data_shape1ax)

            self.net_class1ax = NetFactory.create(net_type1ax)
            self.net1ax = self.net_class1ax(num_classes=self.class_num1ax, w_regularizer=None, b_regularizer=None, name=net_name1ax)

            self.net1ax.set_params(config_net1ax)
            self.predicty1ax = self.net1ax(self.x1ax, is_training=True)
            self.proby1ax = tf.nn.softmax(self.predicty1ax)

            # construct graph for 1st network sagittal
            net_type1sg = config_net1sg['net_type']
            net_name1sg = config_net1sg['net_name']
            self.data_shape1sg = config_net1sg['data_shape']
            self.label_shape1sg = config_net1sg['label_shape']
            class_num1sg = config_net1sg['class_num']

            full_data_shape1sg = [self.batch_size] + self.data_shape1sg
            self.x1sg = tf.placeholder(tf.float32, shape=full_data_shape1sg)

            self.net_class1sg = NetFactory.create(net_type1sg)
            self.net1sg = self.net_class1sg(num_classes=class_num1sg, w_regularizer=None, b_regularizer=None, name=net_name1sg)

            self.net1sg.set_params(config_net1sg)
            self.predicty1sg = self.net1sg(self.x1sg, is_training=True)
            self.proby1sg = tf.nn.softmax(self.predicty1sg)

            # construct graph for 1st network corogal
            net_type1cr = config_net1cr['net_type']
            net_name1cr = config_net1cr['net_name']
            self.data_shape1cr = config_net1cr['data_shape']
            self.label_shape1cr = config_net1cr['label_shape']
            class_num1cr = config_net1cr['class_num']

            full_data_shape1cr = [self.batch_size] + self.data_shape1cr
            self.x1cr = tf.placeholder(tf.float32, shape=full_data_shape1cr)

            self.net_class1cr = NetFactory.create(net_type1cr)
            self.net1cr = self.net_class1cr(num_classes=class_num1cr, w_regularizer=None, b_regularizer=None, name=net_name1cr)

            self.net1cr.set_params(config_net1cr)
            self.predicty1cr = self.net1cr(self.x1cr, is_training=True)
            self.proby1cr = tf.nn.softmax(self.predicty1cr)

        if (self.config_test.get('whole_tumor_only', False) is False):
            # 2.2, networks for tumor core
            if (self.config_net2):
                net_type2 = self.config_net2['net_type']
                net_name2 = self.config_net2['net_name']
                self.data_shape2 = self.config_net2['data_shape']
                self.label_shape2 = self.config_net2['label_shape']
                self.class_num2 = self.config_net2['class_num']

                # construct graph for 2st network
                full_data_shape2 = [self.batch_size] + self.data_shape2
                self.x2 = tf.placeholder(tf.float32, shape=full_data_shape2)
                net_class2 = NetFactory.create(net_type2)
                self.net2 = net_class2(num_classes=self.class_num2, w_regularizer=None,
                                  b_regularizer=None, name=net_name2)
                self.net2.set_params(self.config_net2)
                predicty2 = self.net2(self.x2, is_training=True)
                self.proby2 = tf.nn.softmax(predicty2)
            else:
                config_net2ax = self.config['network2ax']
                config_net2sg = self.config['network2sg']
                config_net2cr = self.config['network2cr']

                # construct graph for 2st network axial
                net_type2ax = config_net2ax['net_type']
                net_name2ax = config_net2ax['net_name']
                self.data_shape2ax = config_net2ax['data_shape']
                self.label_shape2ax = config_net2ax['label_shape']
                self.class_num2ax = config_net2ax['class_num']

                full_data_shape2ax = [self.batch_size] + self.data_shape2ax
                self.x2ax = tf.placeholder(tf.float32, shape=full_data_shape2ax)
                net_class2ax = NetFactory.create(net_type2ax)
                self.net2ax = net_class2ax(num_classes=self.class_num2ax, w_regularizer=None,
                                      b_regularizer=None, name=net_name2ax)
                self.net2ax.set_params(config_net2ax)
                predicty2ax = self.net2ax(self.x2ax, is_training=True)
                self.proby2ax = tf.nn.softmax(predicty2ax)

                # construct graph for 2st network sagittal
                net_type2sg = config_net2sg['net_type']
                net_name2sg = config_net2sg['net_name']
                self.data_shape2sg = config_net2sg['data_shape']
                self.label_shape2sg = config_net2sg['label_shape']
                class_num2sg = config_net2sg['class_num']

                full_data_shape2sg = [self.batch_size] + self.data_shape2sg
                self.x2sg = tf.placeholder(tf.float32, shape=full_data_shape2sg)
                net_class2sg = NetFactory.create(net_type2sg)
                self.net2sg = net_class2sg(num_classes=class_num2sg, w_regularizer=None,
                                      b_regularizer=None, name=net_name2sg)
                self.net2sg.set_params(config_net2sg)
                predicty2sg = self.net2sg(self.x2sg, is_training=True)
                self.proby2sg = tf.nn.softmax(predicty2sg)

                # construct graph for 2st network corogal
                net_type2cr = config_net2cr['net_type']
                net_name2cr = config_net2cr['net_name']
                self.data_shape2cr = config_net2cr['data_shape']
                self.label_shape2cr = config_net2cr['label_shape']
                class_num2cr = config_net2cr['class_num']

                full_data_shape2cr = [self.batch_size] + self.data_shape2cr
                self.x2cr = tf.placeholder(tf.float32, shape=full_data_shape2cr)
                net_class2cr = NetFactory.create(net_type2cr)
                self.net2cr = net_class2cr(num_classes=class_num2cr, w_regularizer=None,
                                      b_regularizer=None, name=net_name2cr)
                self.net2cr.set_params(config_net2cr)
                predicty2cr = self.net2cr(self.x2cr, is_training=True)
                self.proby2cr = tf.nn.softmax(predicty2cr)

            # 2.3, networks for enhanced tumor
            if (self.config_net3):
                net_type3 = self.config_net3['net_type']
                net_name3 = self.config_net3['net_name']
                self.data_shape3 = self.config_net3['data_shape']
                self.label_shape3 = self.config_net3['label_shape']
                self.class_num3 = self.config_net3['class_num']

                # construct graph for 3st network
                full_data_shape3 = [self.batch_size] + self.data_shape3
                self.x3 = tf.placeholder(tf.float32, shape=full_data_shape3)
                net_class3 = NetFactory.create(net_type3)
                self.net3 = net_class3(num_classes=self.class_num3, w_regularizer=None,
                                  b_regularizer=None, name=net_name3)
                self.net3.set_params(self.config_net3)
                predicty3 = self.net3(self.x3, is_training=True)
                self.proby3 = tf.nn.softmax(predicty3)
            else:
                config_net3ax = self.config['network3ax']
                config_net3sg = self.config['network3sg']
                config_net3cr = self.config['network3cr']

                # construct graph for 3st network axial
                net_type3ax = config_net3ax['net_type']
                net_name3ax = config_net3ax['net_name']
                self.data_shape3ax = config_net3ax['data_shape']
                self.label_shape3ax = config_net3ax['label_shape']
                self.class_num3ax = config_net3ax['class_num']

                full_data_shape3ax = [self.batch_size] + self.data_shape3ax
                self.x3ax = tf.placeholder(tf.float32, shape=full_data_shape3ax)
                net_class3ax = NetFactory.create(net_type3ax)
                self.net3ax = net_class3ax(num_classes=self.class_num3ax, w_regularizer=None,
                                      b_regularizer=None, name=net_name3ax)
                self.net3ax.set_params(config_net3ax)
                predicty3ax = self.net3ax(self.x3ax, is_training=True)
                self.proby3ax = tf.nn.softmax(predicty3ax)

                # construct graph for 3st network sagittal
                net_type3sg = config_net3sg['net_type']
                net_name3sg = config_net3sg['net_name']
                self.data_shape3sg = config_net3sg['data_shape']
                self.label_shape3sg = config_net3sg['label_shape']
                class_num3sg = config_net3sg['class_num']
                # construct graph for 3st network
                full_data_shape3sg = [self.batch_size] + self.data_shape3sg
                self.x3sg = tf.placeholder(tf.float32, shape=full_data_shape3sg)
                net_class3sg = NetFactory.create(net_type3sg)
                self.net3sg = net_class3sg(num_classes=class_num3sg, w_regularizer=None,
                                      b_regularizer=None, name=net_name3sg)
                self.net3sg.set_params(config_net3sg)
                predicty3sg = self.net3sg(self.x3sg, is_training=True)
                self.proby3sg = tf.nn.softmax(predicty3sg)

                # construct graph for 3st network corogal
                net_type3cr = config_net3cr['net_type']
                net_name3cr = config_net3cr['net_name']
                self.data_shape3cr = config_net3cr['data_shape']
                self.label_shape3cr = config_net3cr['label_shape']
                class_num3cr = config_net3cr['class_num']
                # construct graph for 3st network
                full_data_shape3cr = [self.batch_size] + self.data_shape3cr
                self.x3cr = tf.placeholder(tf.float32, shape=full_data_shape3cr)
                net_class3cr = NetFactory.create(net_type3cr)
                self.net3cr = net_class3cr(num_classes=class_num3cr, w_regularizer=None,
                                      b_regularizer=None, name=net_name3cr)
                self.net3cr.set_params(config_net3cr)
                predicty3cr = self.net3cr(self.x3cr, is_training=True)
                self.proby3cr = tf.nn.softmax(predicty3cr)

        # 3, create session and load trained models
        self.all_vars = tf.global_variables()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if (self.config_net1):
            self.net1_vars = [x for x in self.all_vars if x.name[0:len(net_name1) + 1] == net_name1 + '/']
            self.saver1 = tf.train.Saver(self.net1_vars)
            self.saver1.restore(self.sess, self.config_net1['model_file'])
        else:
            self.net1ax_vars = [x for x in self.all_vars if x.name[0:len(net_name1ax) + 1] == net_name1ax + '/']
            self.saver1ax = tf.train.Saver(self.net1ax_vars)
            self.saver1ax.restore(self.sess, config_net1ax['model_file'])
            self.net1sg_vars = [x for x in self.all_vars if x.name[0:len(net_name1sg) + 1] == net_name1sg + '/']
            self.saver1sg = tf.train.Saver(self.net1sg_vars)
            self.saver1sg.restore(self.sess, config_net1sg['model_file'])
            self.net1cr_vars = [x for x in self.all_vars if x.name[0:len(net_name1cr) + 1] == net_name1cr + '/']
            self.saver1cr = tf.train.Saver(self.net1cr_vars)
            self.saver1cr.restore(self.sess, config_net1cr['model_file'])

        if (self.config_test.get('whole_tumor_only', False) is False):
            if (self.config_net2):
                self.net2_vars = [x for x in self.all_vars if x.name[0:len(net_name2) + 1] == net_name2 + '/']
                self.saver2 = tf.train.Saver(self.net2_vars)
                self.saver2.restore(self.sess, self.config_net2['model_file'])
            else:
                self.net2ax_vars = [x for x in self.all_vars if x.name[0:len(net_name2ax) + 1] == net_name2ax + '/']
                self.saver2ax = tf.train.Saver(self.net2ax_vars)
                self.saver2ax.restore(self.sess, config_net2ax['model_file'])
                self.net2sg_vars = [x for x in self.all_vars if x.name[0:len(net_name2sg) + 1] == net_name2sg + '/']
                self.saver2sg = tf.train.Saver(self.net2sg_vars)
                self.saver2sg.restore(self.sess, config_net2sg['model_file'])
                self.net2cr_vars = [x for x in self.all_vars if x.name[0:len(net_name2cr) + 1] == net_name2cr + '/']
                self.saver2cr = tf.train.Saver(self.net2cr_vars)
                self.saver2cr.restore(self.sess, config_net2cr['model_file'])

            if (self.config_net3):
                self.net3_vars = [x for x in self.all_vars if x.name[0:len(net_name3) + 1] == net_name3 + '/']
                self.saver3 = tf.train.Saver(self.net3_vars)
                self.saver3.restore(self.sess, self.config_net3['model_file'])
            else:
                self.net3ax_vars = [x for x in self.all_vars if x.name[0:len(net_name3ax) + 1] == net_name3ax + '/']
                self.saver3ax = tf.train.Saver(self.net3ax_vars)
                self.saver3ax.restore(self.sess, config_net3ax['model_file'])
                self.net3sg_vars = [x for x in self.all_vars if x.name[0:len(net_name3sg) + 1] == net_name3sg + '/']
                self.saver3sg = tf.train.Saver(self.net3sg_vars)
                self.saver3sg.restore(self.sess, config_net3sg['model_file'])
                self.net3cr_vars = [x for x in self.all_vars if x.name[0:len(net_name3cr) + 1] == net_name3cr + '/']
                self.saver3cr = tf.train.Saver(self.net3cr_vars)
                self.saver3cr.restore(self.sess, config_net3cr['model_file'])

    def parse_request(self, request):
        savepath = tempfile.gettempdir()

        uid = uuid.uuid4()

        mha_flair = str(uid) + '_flair.mha'

        tmp_filename_flair = os.path.join(savepath, mha_flair)

        mha_t1 = str(uid) + '_t1.mha'

        tmp_filename_t1 = os.path.join(savepath, mha_t1)

        mha_t1ce = str(uid) + '_t1ce.mha'

        tmp_filename_t1ce = os.path.join(savepath, mha_t1ce)

        mha_t2 = str(uid) + '_t2.mha'

        tmp_filename_t2 = os.path.join(savepath, mha_t2)

        with open(tmp_filename_flair, 'wb') as f:
            f.write(request.args['Flair'][0])

        with open(tmp_filename_t1, 'wb') as f:
            f.write(request.args['T1'][0])

        with open(tmp_filename_t1ce, 'wb') as f:
            f.write(request.args['T1ce'][0])

        with open(tmp_filename_t2, 'wb') as f:
            f.write(request.args['T2'][0])

        data = {
            'flair': tmp_filename_flair,
            't1': tmp_filename_t1,
            't1ce': tmp_filename_t1ce,
            't2': tmp_filename_t2,
            'uids': uid,
        }

        return data

    def do_inference(self, data):

        dataloader = DataLoaderServing(self.config_data)
        dataloader.load_data([data['flair'], data['t1'], data['t1ce'], data['t2']])
        image_num = dataloader.get_total_image_number()

        # 5, start to test
        test_slice_direction = self.config_test.get('test_slice_direction', 'all')
        save_folder = tempfile.gettempdir()
        test_time = []
        struct = ndimage.generate_binary_structure(3, 2)
        margin = self.config_test.get('roi_patch_margin', 5)

        for i in range(image_num):
            [temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
            t0 = time.time()
            # 5.1, test of 1st network
            if (self.config_net1):
                data_shapes = [self.data_shape1[:-1], self.data_shape1[:-1], self.data_shape1[:-1]]
                label_shapes = [self.label_shape1[:-1], self.label_shape1[:-1], self.label_shape1[:-1]]
                nets = [self.net1, self.net1, self.net1]
                outputs = [self.proby1, self.proby1, self.proby1]
                inputs = [self.x1, self.x1, self.x1]
                class_num = self.class_num1
            else:
                data_shapes = [self.data_shape1ax[:-1], self.data_shape1sg[:-1], self.data_shape1cr[:-1]]
                label_shapes = [self.label_shape1ax[:-1], self.label_shape1sg[:-1], self.label_shape1cr[:-1]]
                nets = [self.net1ax, self.net1sg, self.net1cr]
                outputs = [self.proby1ax, self.proby1sg, self.proby1cr]
                inputs = [self.x1ax, self.x1sg, self.x1cr]
                class_num = self.class_num1ax
            prob1 = test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, self.data_shape1ax[-1],
                                                             class_num,
                                                             self.batch_size, self.sess, nets, outputs, inputs, shape_mode=2)
            pred1 = np.asarray(np.argmax(prob1, axis=3), np.uint16)
            pred1 = pred1 * temp_weight

            wt_threshold = 2000
            if (self.config_test.get('whole_tumor_only', False) is True):
                pred1_lc = ndimage.morphology.binary_closing(pred1, structure=struct)
                pred1_lc = get_largest_two_component(pred1_lc, False, wt_threshold)
                out_label = pred1_lc
            else:
                # 5.2, test of 2nd network
                if (pred1.sum() == 0):
                    print('net1 output is null', temp_name)
                    bbox1 = get_ND_bounding_box(temp_imgs[0] > 0, margin)
                else:
                    pred1_lc = ndimage.morphology.binary_closing(pred1, structure=struct)
                    pred1_lc = get_largest_two_component(pred1_lc, False, wt_threshold)
                    bbox1 = get_ND_bounding_box(pred1_lc, margin)
                sub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox1[0], bbox1[1]) for one_img in temp_imgs]
                sub_weight = crop_ND_volume_with_bounding_box(temp_weight, bbox1[0], bbox1[1])

                if (self.config_net2):
                    data_shapes = [self.data_shape2[:-1], self.data_shape2[:-1], self.data_shape2[:-1]]
                    label_shapes = [self.label_shape2[:-1], self.label_shape2[:-1], self.label_shape2[:-1]]
                    nets = [self.net2, self.net2, self.net2]
                    outputs = [self.proby2, self.proby2, self.proby2]
                    inputs = [self.x2, self.x2, self.x2]
                    class_num = self.class_num2
                else:
                    data_shapes = [self.data_shape2ax[:-1], self.data_shape2sg[:-1], self.data_shape2cr[:-1]]
                    label_shapes = [self.label_shape2ax[:-1], self.label_shape2sg[:-1], self.label_shape2cr[:-1]]
                    nets = [self.net2ax, self.net2sg, self.net2cr]
                    outputs = [self.proby2ax, self.proby2sg, self.proby2cr]
                    inputs = [self.x2ax, self.x2sg, self.x2cr]
                    class_num = self.class_num2ax
                prob2 = test_one_image_three_nets_adaptive_shape(sub_imgs, data_shapes, label_shapes, self.data_shape2ax[-1],
                                                                 class_num, self.batch_size, self.sess, nets, outputs, inputs,
                                                                 shape_mode=1)
                pred2 = np.asarray(np.argmax(prob2, axis=3), np.uint16)
                pred2 = pred2 * sub_weight

                # 5.3, test of 3rd network
                if (pred2.sum() == 0):
                    [roid, roih, roiw] = sub_imgs[0].shape
                    bbox2 = [[0, 0, 0], [roid - 1, roih - 1, roiw - 1]]
                    subsub_imgs = sub_imgs
                    subsub_weight = sub_weight
                else:
                    pred2_lc = ndimage.morphology.binary_closing(pred2, structure=struct)
                    pred2_lc = get_largest_two_component(pred2_lc)
                    bbox2 = get_ND_bounding_box(pred2_lc, margin)
                    subsub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox2[0], bbox2[1]) for one_img in
                                   sub_imgs]
                    subsub_weight = crop_ND_volume_with_bounding_box(sub_weight, bbox2[0], bbox2[1])

                if (self.config_net3):
                    data_shapes = [self.data_shape3[:-1], self.data_shape3[:-1], self.data_shape3[:-1]]
                    label_shapes = [self.label_shape3[:-1], self.label_shape3[:-1], self.label_shape3[:-1]]
                    nets = [self.net3, self.net3, self.net3]
                    outputs = [self.proby3, self.proby3, self.proby3]
                    inputs = [self.x3, self.x3, self.x3]
                    class_num = self.class_num3
                else:
                    data_shapes = [self.data_shape3ax[:-1], self.data_shape3sg[:-1], self.data_shape3cr[:-1]]
                    label_shapes = [self.label_shape3ax[:-1], self.label_shape3sg[:-1], self.label_shape3cr[:-1]]
                    nets = [self.net3ax, self.net3sg, self.net3cr]
                    outputs = [self.proby3ax, self.proby3sg, self.proby3cr]
                    inputs = [self.x3ax, self.x3sg, self.x3cr]
                    class_num = self.class_num3ax

                prob3 = test_one_image_three_nets_adaptive_shape(subsub_imgs, data_shapes, label_shapes,
                                                                 self.data_shape3ax[-1],
                                                                 class_num, self.batch_size, self.sess, nets, outputs, inputs,
                                                                 shape_mode=1)

                pred3 = np.asarray(np.argmax(prob3, axis=3), np.uint16)
                pred3 = pred3 * subsub_weight

                # 5.4, fuse results at 3 levels
                # convert subsub_label to full size (non-enhanced)
                label3_roi = np.zeros_like(pred2)
                label3_roi = set_ND_volume_roi_with_bounding_box_range(label3_roi, bbox2[0], bbox2[1], pred3)
                label3 = np.zeros_like(pred1)
                label3 = set_ND_volume_roi_with_bounding_box_range(label3, bbox1[0], bbox1[1], label3_roi)

                label2 = np.zeros_like(pred1)
                label2 = set_ND_volume_roi_with_bounding_box_range(label2, bbox1[0], bbox1[1], pred2)

                label1_mask = (pred1 + label2 + label3) > 0
                label1_mask = ndimage.morphology.binary_closing(label1_mask, structure=struct)
                label1_mask = get_largest_two_component(label1_mask, False, wt_threshold)
                label1 = pred1 * label1_mask

                label2_3_mask = (label2 + label3) > 0
                label2_3_mask = label2_3_mask * label1_mask
                label2_3_mask = ndimage.morphology.binary_closing(label2_3_mask, structure=struct)
                label2_3_mask = remove_external_core(label1, label2_3_mask)

                if (label2_3_mask.sum() > 0):
                    label2_3_mask = get_largest_two_component(label2_3_mask)

                label1 = (label1 + label2_3_mask) > 0
                label2 = label2_3_mask
                label3 = label2 * label3
                vox_3 = np.asarray(label3 > 0, np.float32).sum()

                if (0 < vox_3 and vox_3 < 30):
                    label3 = np.zeros_like(label2)

                # 5.5, convert label and save output
                out_label = label1 * 2
                if ('Flair' in self.config_data['modality_postfix'] and 'mha' in self.config_data['file_postfix']):
                    out_label[label2 > 0] = 3
                    out_label[label3 == 1] = 1
                    out_label[label3 == 2] = 4
                elif ('flair' in self.config_data['modality_postfix'] and 'nii' in self.config_data['file_postfix']):
                    out_label[label2 > 0] = 1
                    out_label[label3 > 0] = 4
                out_label = np.asarray(out_label, np.int16)

            test_time.append(time.time() - t0)
            final_label = np.zeros(temp_size, np.int16)
            final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)

            temp_name = save_folder + "/{0:}_seg.mha".format(img_names[0])

            save_array_as_nifty_volume(final_label, temp_name, img_names[0])

            data['saved_file'] = temp_name

            print(temp_name)
        return data

    def prepare_response(self, result):
        tmp_segmentation_nii = result['saved_file']

        with open(tmp_segmentation_nii, 'rb') as f:
            vol_string = base64.encodestring(f.read())

        package = [  # THIS defines the return interface of this service
            {'type': 'LabelVolume', 'content': vol_string, 'label': ''},
        ]

        # os.remove(tmp_segmentation_nii)

        return package


@click.command()
@click.option('--api_key', default='')
@click.option('--port', default=9000)
def start_prediction_service(port, api_key):
    params = {
        'port': port,
        'api_key': api_key,
        'modality': 'MRI',
        'anatomy': 'Brain',
        'task': 'Segmentation',
        'description': DESCRIPTION,
        'volume_resolution': 'unk',
        'volume_size': 'unk',
        'name': 'Brain Tumor Segmentation',
    }

    service = Brats17(
        './config15/test_all_class.txt',
        params=params,
        data_read_pipeline=dummy_data_pipeline,
        data_write_pipeline=dummy_data_pipeline,
        image_field='images',
        segmentation_field='label',
        port=9002,
    )

    if api_key is not '':
        service.add_announcement_looping_call()

    service.run()


cli.add_command(start_prediction_service)


if __name__ == '__main__':
    cli()
