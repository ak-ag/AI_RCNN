"""

"""

import logging

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import time
import DCSCN
from helper import args, utilty as util

args.flags.DEFINE_boolean("save_results", True, "Save result, bicubic and loss images.")
args.flags.DEFINE_boolean("compute_bicubic", False, "Compute bicubic performance.")

FLAGS = args.get()


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    if (FLAGS.frozenInference):
        model.load_graph(FLAGS.frozen_graph_path)
        model.build_summary_saver(with_saver=False) # no need because we are not saving any variables
    else:
        model.build_graph()
        model.build_summary_saver()
    model.init_all_variables()

    if FLAGS.test_dataset == "all":
        test_list = ['set5', 'set14', 'bsd100']
    else:
        test_list = [FLAGS.test_dataset]

    for i in range(FLAGS.tests):
        if (not FLAGS.frozenInference):
            model.load_model(FLAGS.load_model_name, trial=i, output_log=True if FLAGS.tests > 1 else False)

        if FLAGS.compute_bicubic:
            for test_data in test_list:
                print(test_data)
                evaluate_bicubic(model, test_data)

        for test_data in test_list:
            evaluate_model(model, test_data)


def evaluate_bicubic(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
    total_psnr = total_ssim = 0

    for filename in test_filenames:
        psnr, ssim = model.evaluate_bicubic(filename, print_console=False)
        total_psnr += psnr
        total_ssim += ssim
    print("Total psnr: ",total_psnr,"Total ssim:",total_ssim)
    logging.info("Bicubic Average [%s] PSNR:%f, SSIM:%f" % (
        test_data, total_psnr / len(test_filenames), total_ssim / len(test_filenames)))


def evaluate_model(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.data_dir + "/" + test_data)
    total_psnr = total_ssim = total_time = 0

    for filename in test_filenames:
        start = time.time()
        if FLAGS.save_results:
            psnr, ssim = model.do_for_evaluate_with_output(filename, output_directory=FLAGS.output_dir,
                                                           print_console=False)
        else:
            psnr, ssim = model.do_for_evaluate(filename, print_console=False)
        end = time.time()
        elapsed_time = end - start
        total_psnr += psnr
        total_ssim += ssim
        total_time += elapsed_time
    print("Total psnr: ",total_psnr,"Total ssim:",total_ssim)


    logging.info("Model Average [%s] PSNR:%f, SSIM:%f, Time (s): %f" % (
        test_data, total_psnr / len(test_filenames), total_ssim / len(test_filenames), total_time / len(test_filenames)))


if __name__ == '__main__':
    tf.app.run()
