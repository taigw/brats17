import tensorflow as tf

def rename(checkpoint_from, checkpoint_to, replace_from, replace_to):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_from)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_from):
            print(var_name)
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_from, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            var = tf.Variable(var, name=new_name)

        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_to)

if __name__ == '__main__':
    net_name   = ['wt', 'tc', 'en']
    net_name_c = ['WT', 'TC', 'EN']
    num_pretrain = [10000, 20000, 20000]
    for i in range(3):
        for view in ['sg', 'cr']:
            checkpoint_from = "model15/msnet_{0:}32_{1:}.ckpt".format(net_name[i], num_pretrain[i])
            checkpoint_to   = "model15/msnet_{0:}32{1:}_init".format(net_name[i], view)
            replace_from   = "MSNet_{0:}32".format(net_name_c[i])
            replace_to     = "MSNet_{0:}32{1:}".format(net_name_c[i], view)
            rename(checkpoint_from, checkpoint_to, replace_from, replace_to)

