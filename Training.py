import tensorflow as tf
import VAE


# Train the VAE
def train_sess(img, saving_model, train_init, no_epochs):
    # Set up the VAE
    optimizer, loss = VAE.run_vae(img)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(no_epochs):
            sess.run(train_init)
            try:
                sess.run(optimizer, feed_dict={VAE.keep_prob: 0.8})
            except tf.errors.OutOfRangeError:
                pass

            if (not i % 200) or (i == no_epochs):
                try:
                    ls = sess.run(loss, feed_dict={VAE.keep_prob: 1.0})
                    print(i, ls)
                except tf.errors.OutOfRangeError:
                    pass
        if saving_model:
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            # Save the Model after Training
            print(saver.save(sess, 'C:/Users/simon/Documents/Uni_Stuttgart/Forschungsarbeit/Code/Models/VAE/VAE_decoder'))
    return ls
