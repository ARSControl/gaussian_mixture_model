#include <iostream>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <vector>
#include <chrono>

#include <gaussian_mixture_model/gaussian_mixture_model.h>


GaussianMixtureModel::GaussianMixtureModel(std::vector<Eigen::VectorXd> means, std::vector<Eigen::MatrixXd> covariances, std::vector<double> weights)
{
    std::cout << "Gaussian Mixture Model constructor called from existing clusters" << std::endl;
    
    // Check input data consistency 
    if (means.size() != covariances.size() || means.size() != weights.size())
    {
        throw std::invalid_argument("Error in GMM definition: size mismatch");
    }

    k_ = means.size();
    dim_ = means[0].size();
    mu_ = means;
    sigma_ = covariances;
    w_ = weights;
    log_likelihood_ = 0.0;
}


GaussianMixtureModel::GaussianMixtureModel(Eigen::MatrixXd samples, int num_components)
{
    std::cout << "Gaussian Mixture Model constructor called. Fitting GMM to data..." << std::endl;
    fitgmm(samples, num_components, 1000, 1e-3, false);
}


GaussianMixtureModel::GaussianMixtureModel(int num_components)
{
    std::cout << "Gaussian Mixture Model constructor called. Crating random GMM..." << std::endl;
    k_ = num_components;
    log_likelihood_ = std::numeric_limits<double>::infinity();

    // Generate random inputs
    double dev = 10.0;
    std::normal_distribution<double> dist_x(0.0, dev);
    std::normal_distribution<double> dist_y(0.0, dev);
    std::normal_distribution<double> dist_theta(0.0, dev);
    std::default_random_engine gen;

    for (int i = 0; i < k_; i++)
    {   
        // Assign equal weight to each component
        w_.push_back(1.0 / k_);

        Eigen::VectorXd q(2);
        q << dist_x(gen), dist_y(gen);
        // mu_[i](0) = dist_x(gen);
        // mu_[i](1) = dist_y(gen);

        mu_.push_back(q);

        Eigen::MatrixXd cov(2,2);
        cov << dev, dev/5.0,
             dev/5.0, dev;
        sigma_.push_back(cov);

    }

}


GaussianMixtureModel::~GaussianMixtureModel()
{
    std::cout << "Gaussian Mixture Model destructor called" << std::endl;
}


std::vector<Eigen::VectorXd> GaussianMixtureModel::getMeans()
{
    return mu_;
}


std::vector<Eigen::MatrixXd> GaussianMixtureModel::getCovariances()
{
    return sigma_;
}


std::vector<double> GaussianMixtureModel::getWeights()
{
    return w_;
}

 
double GaussianMixtureModel::getLogLikelihood()
{
    return log_likelihood_;
}



double GaussianMixtureModel::gauss_pdf_2d(Eigen::VectorXd q, Eigen::VectorXd mean, Eigen::MatrixXd cov)
{
    double det = cov.determinant();
    double inv_sqrt_det = 1.0 / sqrt(det);
    Eigen::VectorXd q_m = q - mean;

    
    double exponent = -0.5 * q_m.transpose() * cov.inverse() * q_m;
    double coeff = inv_sqrt_det / (2.0 * M_PI);
    return coeff * exp(exponent);
}


void GaussianMixtureModel::fitgmm(Eigen::MatrixXd samples, int num_components, int max_iterations = 1000, double tolerance = 1e-3, bool verbose = false)
{
    auto timerstart = std::chrono::high_resolution_clock::now();
    if (verbose)
    {
        std::cout << "Fitting GMM to data..." << std::endl;
    }

    // Set GMM parameters
    dim_ = samples.rows();
    k_ = num_components;
    int n_samples = samples.cols();

    if (verbose)
    {
        std::cout << "Number of samples: " << n_samples << std::endl;
        std::cout << "Number of components: " << k_ << std::endl;
        std::cout << "Dimension of samples: " << dim_ << std::endl;
    }

    // Initialize new GMM parameters
    // std::vector<double> gamma(n_sampels * k_);               // responsibilities
    std::vector<double> gamma(n_samples * k_);               // responsibilities
    // double log_likelihood_old = -std::numeric_limits<double>::infinity();
    double log_likelihood_old = -1.0;
    double log_likelihood_new = 0.0;
    int it = 0;                                // iteration counter

    while (it < max_iterations && abs(log_likelihood_new-log_likelihood_old) > tolerance)
    {
        log_likelihood_old = log_likelihood_new;

        // E-step: compute responsibilities
        for (int i = 0; i < n_samples; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < k_; j++)
            {
                double prob = w_[j] * gauss_pdf_2d(samples.col(i).head(2), mu_[j], sigma_[j]);
                sum += prob;
                gamma[i * k_ + j] = prob;
            }

            for (int j = 0; j < k_; j++)
            {
                gamma[i * k_ + j] /= sum;
            }

            // gamma.row(i) /= sum;
        }

        // M-step: update parameters
        for (int j = 0; j < k_; j++)
        {
            double sum_gamma = 0.0;
            double sum_x1 = 0.0;
            double sum_x2 = 0.0;
            double sum_x1_squared = 0.0;
            double sum_x2_squared = 0.0;
            double sum_x1_x2 = 0.0;

            for (int i = 0; i < n_samples; i++)
            {
                sum_gamma += gamma[i * k_ + j];
                sum_x1 += gamma[i * k_ + j] * samples(0,i);
                sum_x2 += gamma[i * k_ + j] * samples(1,i);
                sum_x1_squared += gamma[i * k_ + j] * pow(samples(0,i), 2);
                sum_x2_squared += gamma[i * k_ + j] * pow(samples(1,i), 2);
                sum_x1_x2 += gamma[i * k_ + j] * samples(0,i) * samples(1,i);
            }

            w_[j] = sum_gamma / n_samples;
            mu_[j](0) = sum_x1 / sum_gamma;
            mu_[j](1) = sum_x2 / sum_gamma;
            sigma_[j](0,0) = sum_x1_squared / sum_gamma - pow(mu_[j](0), 2);
            sigma_[j](1,1) = sum_x2_squared / sum_gamma - pow(mu_[j](1), 2);
            sigma_[j](0,1) = sum_x1_x2 / sum_gamma - mu_[j](0) * mu_[j](1);
            sigma_[j](1,0) = sigma_[j](0,1);
        }

        // Compute log-likelihood of data
        log_likelihood_new = 0.0;
        for (int i = 0; i < n_samples; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < k_; j++)
            {
                sum += w_[j] * gauss_pdf_2d(samples.col(i).head(2), mu_[j], sigma_[j]);
            }
            log_likelihood_new += log(sum);
        }

        it++;

        if (verbose)
        {
            std::cout << "Iteration: " << it << std::endl;
            std::cout << "Actual Log likelihood: " << log_likelihood_new << std::endl;
            std::cout << "Difference new vs old log-likelihood: " << log_likelihood_new - log_likelihood_old << std::endl;
        }
    }

    log_likelihood_ = log_likelihood_new;
    auto end = std::chrono::high_resolution_clock::now();

    if (verbose)
    {   
        std::cout << "Total number of iterations: " << it << std::endl;
        std::cout << "Log likelihood: " << log_likelihood_new << std::endl;
        std::cout<<"Computation time for EM: -------------: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - timerstart).count()<<" ms :-------------\n";
    }
}


void GaussianMixtureModel::fitgmm(std::vector<Eigen::VectorXd> samples, int num_components, int max_iterations = 1000, double tolerance = 1e-3, bool verbose = false)
{
    // Rework samples to Eigen::MatrixXd
    int n_samples = samples.size();
    int dim = samples[0].size();
    Eigen::MatrixXd samples_eigen(dim, n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        samples_eigen.col(i) = samples[i];
    }

    // Fit GMM
    fitgmm(samples_eigen, num_components, max_iterations, tolerance, verbose);
}


