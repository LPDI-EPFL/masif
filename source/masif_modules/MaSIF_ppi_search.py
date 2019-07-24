import tensorflow as tf
import numpy as np


class MaSIF_ppi_search:

    """
    The neural network model to classify two patches into binders or not binders. 
    """

    def count_number_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print("Total number parameters: %d" % total_parameters)

    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm

    def build_sparse_matrix_softmax(self, idx_non_zero_values, X, dense_shape_A):
        A = tf.SparseTensorValue(idx_non_zero_values, tf.squeeze(X), dense_shape_A)
        A = tf.sparse_reorder(A)  # n_edges x n_edges
        A = tf.sparse_softmax(A)

        return A

    def compute_initial_coordinates(self):
        range_rho = [0.0, self.max_rho]
        range_theta = [0, 2 * np.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1)
        grid_rho = grid_rho[1:]
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)
        grid_theta = grid_theta[:-1]

        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
        grid_rho_ = (
            grid_rho_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_theta_ = (
            grid_theta_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0)
        coords = coords.T  # every row contains the coordinates of a grid intersection
        print(coords.shape)
        return coords

    def inference(
        self,
        input_feat,
        rho_coords,
        theta_coords,
        mask,
        W_conv,
        b_conv,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
        n_samples = tf.shape(rho_coords)[0]
        n_vertices = tf.shape(rho_coords)[1]
        # n_feat = input_feat.get_shape().as_list()[2]

        all_conv_feat = []
        for k in range(self.n_rotations):
            rho_coords_ = tf.reshape(rho_coords, [-1, 1])  # batch_size*n_vertices
            thetas_coords_ = tf.reshape(theta_coords, [-1, 1])  # batch_size*n_vertices

            thetas_coords_ += k * 2 * np.pi / self.n_rotations
            thetas_coords_ = tf.mod(thetas_coords_, 2 * np.pi)
            rho_coords_ = tf.exp(
                -tf.square(rho_coords_ - mu_rho) / (tf.square(sigma_rho) + eps)
            )
            thetas_coords_ = tf.exp(
                -tf.square(thetas_coords_ - mu_theta) / (tf.square(sigma_theta) + eps)
            )

            gauss_activations = tf.multiply(
                rho_coords_, thetas_coords_
            )  # batch_size*n_vertices, n_gauss
            gauss_activations = tf.reshape(
                gauss_activations, [n_samples, n_vertices, -1]
            )  # batch_size, n_vertices, n_gauss
            gauss_activations = tf.multiply(gauss_activations, mask)
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(gauss_activations, 1, keep_dims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 2
            )  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = tf.expand_dims(
                input_feat, 3
            )  # batch_size, n_vertices, n_feat, 1

            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(gauss_desc, 1)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [n_samples, self.n_thetas * self.n_rhos]
            )  # batch_size, 80

            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, 80
            all_conv_feat.append(conv_feat)
        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(all_conv_feat, 0)
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    # Softmax cross entropy
    def compute_data_loss_cross_entropy(self, pos, neg):
        epsilon = tf.constant(value=0.00001)
        logit = tf.nn.softmax([pos, neg])
        self.softmax_debug = logit
        cross_entropy = -(tf.log(logit[1] + epsilon) - tf.log(logit[0] + epsilon))
        return cross_entropy

    # Data loss
    # Values above 10 are ignored.
    def compute_data_loss(self, pos_thresh=0.0, neg_thresh=10):
        self.global_desc_pos = tf.gather(self.global_desc, tf.range(0, self.n_patches))
        self.global_desc_binder = tf.gather(
            self.global_desc, tf.range(self.n_patches, 2 * self.n_patches)
        )
        self.global_desc_neg = tf.gather(
            self.global_desc, tf.range(2 * self.n_patches, 3 * self.n_patches)
        )
        self.global_desc_neg_2 = tf.gather(
            self.global_desc, tf.range(3 * self.n_patches, 4 * self.n_patches)
        )

        pos_distances = tf.reduce_sum(
            tf.square(self.global_desc_binder - self.global_desc_pos), 1
        )
        neg_distances = tf.reduce_sum(
            tf.square(self.global_desc_neg - self.global_desc_neg_2), 1
        )
        self.score = tf.concat([pos_distances, neg_distances], axis=0)
        pos_distances = tf.nn.relu(
            tf.reduce_sum(tf.square(self.global_desc_binder - self.global_desc_pos), 1)
            - pos_thresh
        )
        neg_distances = tf.nn.relu(
            -tf.reduce_sum(tf.square(self.global_desc_neg - self.global_desc_neg_2), 1)
            + neg_thresh
        )

        pos_mean, pos_std = tf.nn.moments(pos_distances, [0])
        neg_mean, neg_std = tf.nn.moments(neg_distances, [0])
        data_loss = pos_std + neg_std + pos_mean + neg_mean

        return data_loss

    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_gamma=1.0,
        learning_rate=1e-3,
        n_rotations=16,
        idx_gpu="/device:GPU:0",
        feat_mask=[1.0, 1.0, 1.0, 1.0, 1.0],
    ):

        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos

        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))

        with tf.Graph().as_default() as g:
            self.graph = g
            tf.set_random_seed(0)
            with tf.device(idx_gpu):

                initial_coords = self.compute_initial_coordinates()
                mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype(
                    "float32"
                )
                mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype(
                    "float32"
                )
                self.mu_rho = []
                self.mu_theta = []
                self.sigma_rho = []
                self.sigma_theta = []
                for i in range(self.n_feat):
                    self.mu_rho.append(
                        tf.Variable(mu_rho_initial, name="mu_rho_{}".format(i))
                    )  # 1, n_gauss
                    self.mu_theta.append(
                        tf.Variable(mu_theta_initial, name="mu_theta_{}".format(i))
                    )  # 1, n_gauss
                    self.sigma_rho.append(
                        tf.Variable(
                            np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                            name="sigma_rho_{}".format(i),
                        )
                    )  # 1, n_gauss
                    self.sigma_theta.append(
                        tf.Variable(
                            (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                            name="sigma_theta_{}".format(i),
                        )
                    )  # 1, n_gauss

                self.keep_prob = tf.placeholder(tf.float32)
                # **Features for binder should be flipped before feeding to the NN.
                self.rho_coords = tf.placeholder(
                    tf.float32, shape=[None, None, 1]
                )  # batch_size, n_vertices, 1
                self.theta_coords = tf.placeholder(
                    tf.float32, shape=[None, None, 1]
                )  # batch_size, n_vertices, 1
                self.input_feat = tf.placeholder(
                    tf.float32, shape=[None, None, self.n_feat]
                )  # batch_size, n_vertices, n_feat
                self.mask = tf.placeholder(
                    tf.float32, shape=[None, None, 1]
                )  # batch_size, n_vertices, 1

                self.global_desc = []

                # Initialize b_conv for each feature.
                b_conv = []
                for i in range(self.n_feat):
                    b_conv.append(
                        tf.Variable(
                            tf.zeros([self.n_thetas * self.n_rhos]),
                            name="b_conv_{}".format(i),
                        )
                    )
                # Run the inference layer per feature.
                for i in range(self.n_feat):
                    my_input_feat = tf.expand_dims(self.input_feat[:, :, i], 2)

                    W_conv = tf.get_variable(
                        "W_conv_{}".format(i),
                        shape=[
                            self.n_thetas * self.n_rhos,
                            self.n_thetas * self.n_rhos,
                        ],
                        initializer=tf.contrib.layers.xavier_initializer(),
                    )

                    desc = self.inference(
                        my_input_feat,
                        self.rho_coords,
                        self.theta_coords,
                        self.mask,
                        W_conv,
                        b_conv[i],
                        self.mu_rho[i],
                        self.sigma_rho[i],
                        self.mu_theta[i],
                        self.sigma_theta[i],
                    )  # batch_size, n_gauss*1

                    self.global_desc.append(desc)

                # global_desc is [n_feat, batch_size, self.n_thetas*self.n_rhos].
                self.global_desc = tf.stack(self.global_desc, axis=1)  #
                self.global_desc = tf.reshape(
                    self.global_desc, [-1, self.n_thetas * self.n_rhos * self.n_feat]
                )

                # Refine global_desc with a FC layer.
                self.global_desc = tf.contrib.layers.fully_connected(
                    self.global_desc,
                    self.n_thetas * self.n_rhos,
                    activation_fn=tf.identity,
                )  # batch_size, n_thetas

                # compute data loss
                self.n_patches = tf.shape(self.global_desc)[0] // 4
                self.data_loss = self.compute_data_loss()

                # definition of the solver
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate
                ).minimize(self.data_loss)

                self.var_grad = tf.gradients(self.data_loss, tf.trainable_variables())
                # print self.var_grad
                for k in range(len(self.var_grad)):
                    if self.var_grad[k] is None:
                        print(tf.trainable_variables()[k])
                self.norm_grad = self.frobenius_norm(
                    tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0)
                )

                # Create a session for running Ops on the Graph.
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.Session(config=config)
                self.saver = tf.train.Saver()

                # Run the Op to initialize the variables.
                init = tf.global_variables_initializer()
                self.session.run(init)
                self.count_number_parameters()

