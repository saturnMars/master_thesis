import tensorflow as tf

class adVAE(object):

    def __init__(self, height, width, channel, z_dim, mx, mz, leaning_rate=1e-3):

        """The class adVAE conducts constructing adVAE with N steps as below.
        Step 1. Constructing neural network.
        Step 2. Defincing the loss functions.
        Step 3. Splitting the parameters for two training phase.
        Step 4. assign splitted parameters for two optimizer respectively.
        Extra Step. TensorBoard."""

        print("\nInitializing Neural Network...")

        """ -------------------- Step 1 -------------------- """
        self.height, self.width, self.channel = height, width, channel
        self.k_size, self.z_dim = 3, z_dim
        self.mx, self.mz, = mx, mz
        self.leaning_rate = leaning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[])
        self.lambda_T, self.lambda_G = 1e-22, 1e+1

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []
        self.features_r, self.features_f = [], []

        self.z_pack, self.z_T_pack, self.x_r, self.x_Tr, self.z_r_pack, self.z_Tr_pack = \
            self.build_model(input=self.x, ksize=self.k_size)

        """ -------------------- Step 2 -------------------- """
        # Loss of Transformer (T)
        # z_pack is constructed as [0:z, 1:z_mu, 2:z_sigma]
        self.T_term1 = tf.math.log(self.z_T_pack[2] + 1e-12) - tf.math.log(self.z_pack[2] + 1e-12)
        self.T_term2 = (tf.square(self.z_pack[2]) + tf.square(self.z_T_pack[1] - self.z_pack[1])) / (2 * tf.square(self.z_T_pack[2]))
        self.T_term3 = - 0.5
        self.loss_T = tf.compat.v1.reduce_sum(self.T_term1 + self.T_term2 + self.T_term3, axis=(1))

        # Loss of Generator (G)
        self.mse_r = self.mean_square_error(x1=self.x, x2=self.x_r)
        self.kld_r = self.kl_divergence(mu=self.z_r_pack[1], sigma=self.z_r_pack[2])
        self.mse_Tr = self.mean_square_error(x1=self.x_r, x2=self.x_Tr)
        self.kld_Tr = self.kl_divergence(mu=self.z_Tr_pack[1], sigma=self.z_Tr_pack[2])
        self.loss_G_z = self.mse_r + self.kld_r
        self.loss_G_zT = self.max_with_positive_margin(margin=self.mx, loss=self.mse_Tr) + \
            self.max_with_positive_margin(margin=self.mz, loss=self.kld_Tr)
        self.loss_G = self.loss_G_z + self.loss_G_zT

        # Loss of Encoder (E)
        self.kld = self.kl_divergence(mu=self.z_pack[1], sigma=self.z_pack[2])
        self.loss_E = self.kld + self.mse_r + \
            self.max_with_positive_margin(margin=self.mz, loss=self.kld_r) + \
            self.max_with_positive_margin(margin=self.mz, loss=self.kld_Tr)

        self.loss_T_mean = tf.compat.v1.reduce_mean(self.lambda_T * self.loss_T)
        self.loss_G_mean = tf.compat.v1.reduce_mean(self.lambda_G * self.loss_G)
        self.loss_E_mean = tf.compat.v1.reduce_mean(self.loss_E)

        self.loss1 = tf.compat.v1.reduce_mean((self.lambda_T * self.loss_T) + (self.lambda_G * self.loss_G))
        self.loss2 = self.loss_E_mean
        self.loss_tot = self.loss1 + self.loss2 # meaningless but use for confirming loss convergence.

        """ -------------------- Step 3 -------------------- """
        self.vars1, self.vars2 = [], []
        for widx, wname in enumerate(self.w_names):
            if("enc_" in wname):
                self.vars2.append(self.weights[widx])
                self.vars2.append(self.biasis[widx])
            elif(("gen_" in wname) or ("tra_" in wname)):
                self.vars1.append(self.weights[widx])
                self.vars1.append(self.biasis[widx])
            else: pass

        print("\nVariables (T and G)")
        for var in self.vars1: print(var)
        print("\nVariables (E)")
        for var in self.vars2: print(var)

        """ -------------------- Step 4 -------------------- """
        self.optimizer1 = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss1, var_list=self.vars1, name='Adam_T_G')
        self.optimizer2 = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss2, var_list=self.vars2, name='Adam_E')

        """ -------------------- Extra Step -------------------- """
        tf.compat.v1.summary.scalar('loss_T', self.loss_T_mean)
        tf.compat.v1.summary.scalar('loss_G', self.loss_G_mean)
        tf.compat.v1.summary.scalar('loss_E', self.loss_E_mean)
        tf.compat.v1.summary.scalar('loss_tot', self.loss_tot)
        self.summaries = tf.compat.v1.summary.merge_all()

    """ ----------------------------------------------------------------------------
    The functions for measuring losses are defined below this sentence.
    ---------------------------------------------------------------------------- """

    def mean_square_error(self, x1, x2):

        """Measure MSE between x1, x2."""

        data_dim = len(x1.shape)
        if(data_dim == 4):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3))
        elif(data_dim == 3):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2))
        elif(data_dim == 2):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1))
        else:
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2))

    def kl_divergence(self, mu, sigma):

        """Measure KL-divergence between N(mu, sigma^2) and N(0, 1)."""

        return 0.5 * tf.compat.v1.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.math.log(tf.square(sigma) + 1e-12) - 1, axis=(1))

    def max_with_positive_margin(self, margin, loss):

        """Measure marginal loss."""

        return tf.compat.v1.math.maximum(x=tf.compat.v1.zeros_like(loss), y=(-loss + margin))

    """ ----------------------------------------------------------------------------
    The functions for building neural network are defined below this sentence.
    ---------------------------------------------------------------------------- """

    def build_model(self, input, ksize=3):

        """Build adVAE structure using three functions 'encoder', 'transformer', and 'generator'.
        The function 'generator' is same as decoder in VAE structure."""

        with tf.compat.v1.variable_scope('encoder') as scope_enc:
            z, z_mu, z_sigma = self.encoder(input=input, ksize=ksize, name="enc")
            z_pack = [z, z_mu, z_sigma]

        with tf.compat.v1.variable_scope('transformer') as scope_tra:
            z_T, z_mu_T, z_sigma_T = self.transformer(input=z, name="tra")
            z_T_pack = [z_T, z_mu_T, z_sigma_T]

        with tf.compat.v1.variable_scope('generator') as scope_gen:
            x_r = self.generator(input=z, ksize=ksize, name="gen")
            x_Tr = self.generator(input=z_T, ksize=ksize, name="gen")

        with tf.compat.v1.variable_scope(scope_enc, reuse=True):
            z_r, z_mu_r, z_sigma_r = self.encoder(input=x_r, ksize=ksize, name="enc")
            z_r_pack = [z_r, z_mu_r, z_sigma_r]
            z_Tr, z_mu_Tr, z_sigma_Tr = self.encoder(input=x_Tr, ksize=ksize, name="enc")
            z_Tr_pack = [z_Tr, z_mu_Tr, z_sigma_Tr]

        return z_pack, z_T_pack, x_r, x_Tr, z_r_pack, z_Tr_pack

    def encoder(self, input, ksize=3, name="enc"):

        """The function for generating encoder. Basically the purpose of this fuction is encoding
        the input data x to latent vector z. Each parameter for each layer is generated with its
        own name. However, when who try to generate the parameter using existed name, the existing
        parameter will be used in that situation. The parameter sharing is conducted via above
        property."""

        print("\nEncoder-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 1, 16], activation="elu", name="%s_conv1_1" %(name))
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="%s_conv1_2" %(name))
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="%s_max_pool1" %(name))
        self.conv_shapes.append(conv1_2.shape)

        print("Encoder-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="elu", name="%s_conv2_1" %(name))
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="%s_conv2_2" %(name))
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="%s_max_pool2" %(name))
        self.conv_shapes.append(conv2_2.shape)

        print("Encoder-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="elu", name="%s_conv3_1" %(name))
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="%s_conv3_2" %(name))
        self.conv_shapes.append(conv3_2.shape)

        print("Encoder-Dense")
        self.fc_shapes.append(conv3_2.shape)
        [n, h, w, c] = self.fc_shapes[0]
        fulcon_in = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="%s_fulcon_in" %(name))
        fulcon1 = self.fully_connected(input=fulcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="elu", name="%s_fullcon1" %(name))

        z_params = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=self.z_dim*2, activation="None", name="%s_z_param" %(name))
        z_mu, z_sigma = self.split_z(z=z_params)
        z = self.sample_z(mu=z_mu, sigma=z_sigma) # reparameterization trick

        return z, z_mu, z_sigma

    def transformer(self, input, name="tra"):

        """This function has same context as function 'encoder'. But, the purpose of this function
        is transforming normal latent vector to unkown anomalous latent vector."""

        print("\nTransformer-Dense")
        z_params = self.fully_connected(input=input, num_inputs=self.z_dim, \
            num_outputs=self.z_dim*2, activation="elu", name="%s_fullcon1" %(name))
        z_mu, z_sigma = self.split_z(z=z_params)
        z = self.sample_z(mu=z_mu, sigma=z_sigma) # reparameterization trick

        return z, z_mu, z_sigma

    def generator(self, input, ksize=3, name="dec"):

        """This function has same context as function 'encoder'. But, the purpose of this function
        is restoring the input x from latent vector z."""

        print("\nGenerator-Dense")
        [n, h, w, c] = self.fc_shapes[0]
        fulcon2 = self.fully_connected(input=input, num_inputs=int(self.z_dim), \
            num_outputs=512, activation="elu", name="%s_fullcon2" %(name))
        fulcon3 = self.fully_connected(input=fulcon2, num_inputs=int(fulcon2.shape[1]), \
            num_outputs=int(h*w*c), activation="elu", name="%s_fullcon3" %(name))
        fulcon_out = tf.compat.v1.reshape(fulcon3, shape=[self.batch_size, h, w, c], name="%s_fulcon_out" %(name))

        print("Generator-1")
        convt1_1 = self.conv2d(input=fulcon_out, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="%s_convt1_1" %(name))
        convt1_2 = self.conv2d(input=convt1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="%s_convt1_2" %(name))

        print("Generator-2")
        [n, h, w, c] = self.conv_shapes[-2]
        convt2_1 = self.conv2d_transpose(input=convt1_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 32, 64], \
            dilations=[1, 1, 1, 1], activation="elu", name="%s_convt2_1" %(name))
        convt2_2 = self.conv2d(input=convt2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="%s_convt2_2" %(name))

        print("Generator-3")
        [n, h, w, c] = self.conv_shapes[-3]
        convt3_1 = self.conv2d_transpose(input=convt2_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 16, 32], \
            dilations=[1, 1, 1, 1], activation="elu", name="%s_convt3_1" %(name))
        convt3_2 = self.conv2d(input=convt3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="%s_convt3_2" %(name))
        convt3_3 = self.conv2d(input=convt3_2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 1], activation="sigmoid", name="%s_convt3_3" %(name))

        return convt3_3

    def split_z(self, z):

        z_mu = z[:, :self.z_dim]
        # z_mu = tf.compat.v1.clip_by_value(z_mu, -3+(1e-12), 3-(1e-12))
        z_sigma = z[:, self.z_dim:]
        z_sigma = tf.compat.v1.clip_by_value(z_sigma, 1e-12, 1-(1e-12))

        return z_mu, z_sigma

    def sample_z(self, mu, sigma):

        # default of tf.random.normal: mean=0.0, stddev=1.0
        epsilon = tf.random.normal(tf.shape(mu), dtype=tf.float32)
        sample = mu + (sigma * epsilon)

        return sample

    """ ----------------------------------------------------------------------------
    The functions for constructing layers are defined below this sentence.
    ---------------------------------------------------------------------------- """

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.compat.v1.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.compat.v1.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        """The function for construct the variable bank. All the variables (parameters) has its own name.
        Thus, if who try to make variable with extisted variable name, this function will return existing
        variable without generating new variable. This function is useful for parameter sharing."""

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        """This function is customized 2D-convolution. Parameter sharing or someting else can be adjusted
        in this function. The parameter W(weight) and b(bias) is generated by function 'variable_maker'
        firstly. Then, the convolutional calculation with W and adding b are conducted."""

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        """This function has same context as function 'conv2d'."""

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-2]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d_transpose(
            value=input,
            filter=weight,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv-Tr", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        """This function has same context as function 'conv2d'. The only difference is that 'conv2d' uses
        convolutional calculation and this function uses inner-product calculation."""

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=[num_inputs, num_outputs], name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[num_outputs], name='%s_b' %(name))

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)