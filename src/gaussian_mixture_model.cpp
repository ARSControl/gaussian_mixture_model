#include <iostream>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <vector>
#include <chrono>
#include <math.h>

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

GaussianMixtureModel::GaussianMixtureModel()
{
    std::cout << "Gaussian Mixture Model constructor called. Creating empty GMM..." << std::endl;
    log_likelihood_ = -1.0;
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

void GaussianMixtureModel::setMeans(std::vector<Eigen::VectorXd> means)
{
    mu_ = means;
    k_ = means.size();
}

void GaussianMixtureModel::setCovariances(std::vector<Eigen::MatrixXd> covariances)
{
    sigma_ = covariances;
}

void GaussianMixtureModel::setWeights(std::vector<double> weights)
{
    w_ = weights;
}

bool GaussianMixtureModel::check()
{
    if (mu_.size() != sigma_.size() || mu_.size() != w_.size() || mu_.size() != k_)
    {
        std::cout << "Error in GMM definition: element number mismatch" << std::endl;
        return false;
    } else if (mu_[0].size() != sigma_[0].rows() || mu_[0].size() != sigma_[0].cols())
    {
        std::cout << "Error in GMM definition: dimension mismatch" << std::endl;
        return false;
    } else
    {
        return true;
    }
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
    Eigen::MatrixXd gamma(n_samples, k_);               // responsibilities
    double log_likelihood_old = -1.0;
    double log_likelihood_new = 0.0;
    int it = 0;                                // iteration counter

    while (it < max_iterations && abs(log_likelihood_new-log_likelihood_old) > tolerance)
    {
        log_likelihood_old = log_likelihood_new;

        if (verbose) {std::cout << "Starting E-step..." << std::endl;}

        // E-step: compute responsibilities
        for (int i = 0; i < n_samples; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < k_; j++)
            {
                double prob = w_[j] * gauss_pdf_2d(samples.col(i), mu_[j], sigma_[j]);
                sum += prob;
                gamma(i,j) = prob;
            }

            gamma.row(i) /= sum;
        }

        if (verbose) {std::cout << "E-step completed. Starting M-step..." << std::endl;}

        // M-step: update parameters
        for (int j = 0; j < k_; j++)
        {
            double sum_gamma = gamma.col(j).sum();
            Eigen::VectorXd weighted_sum = samples * gamma.col(j);
            if (verbose)
            {
                std::cout << "Sum of gamma: " << sum_gamma << std::endl;
                std::cout << "Weighted sum: " << weighted_sum.transpose() << std::endl;
            }

            if (!isnan(weighted_sum(0)) && !isnan(weighted_sum(1)) && !isnan(sum_gamma))
            {
                w_[j] = sum_gamma / n_samples;
                mu_[j] = weighted_sum / sum_gamma;
            }

            Eigen::VectorXd sum_x_squared = samples.cwiseAbs2() * gamma.col(j);      // vector 2x1
            if (verbose)
            {
                std::cout << "Sum of x squared: " << sum_x_squared.transpose() << std::endl;
            }

            double sum_x1_x2 = samples.row(0).cwiseProduct(samples.row(1)) * gamma.col(j);  // scalar
            if (verbose)
            {
                std::cout << "Sum of x1*x2: " << sum_x1_x2 << std::endl;
            }

            sigma_[j](0,0) = sum_x_squared(0) / sum_gamma - pow(mu_[j](0), 2);
            sigma_[j](1,1) = sum_x_squared(1) / sum_gamma - pow(mu_[j](1), 2);
            sigma_[j](0,1) = sum_x1_x2 / sum_gamma - mu_[j](0) * mu_[j](1);
            sigma_[j](1,0) = sigma_[j](0,1);
        }

        if(verbose) {std::cout << "M-step completed. Computing log-likelihood..." << std::endl;}

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
    Eigen::MatrixXd samples_eigen(2, n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        samples_eigen.col(i) = samples[i].head(2);
    }

    // Fit GMM
    fitgmm(samples_eigen, num_components, max_iterations, tolerance, verbose);
}



std::vector<Eigen::VectorXd> GaussianMixtureModel::drawSamples(int num_samples)
{
    std::cout << "Drawing samples..." << std::endl;
    std::default_random_engine gen;
    // std::vector<double> weights;
    std::vector<Eigen::VectorXd> resampled_particles;
    std::vector<Eigen::VectorXd> init_particles;

    // std::cout << "Weights: " << w_ << std::endl;

    // for (int i = 0; i < w_.size(); i++)
    // {
    //     weights.push_back(w_(i));
    //     // init_particles.push_back(particles_.col(i));
    // }

    std::discrete_distribution<int> distribution(w_.begin(), w_.end());

    for (int i = 0; i < num_samples; i++)
    {
        resampled_particles.push_back(init_particles[distribution(gen)]);
    }

    return resampled_particles;
}
