#ifndef GAUSSIAN_MIXTURE_MODEL_H
#define GAUSSIAN_MIXTURE_MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>


class GaussianMixtureModel
{
    private:
        int k_;                                 // number of components
        int dim_;                               // dimension of data
        std::vector<Eigen::VectorXd> mu_;       // mean vectors
        std::vector<Eigen::MatrixXd> sigma_;    // covariance matrices
        std::vector<double> w_;                 // weights of components
        double log_likelihood_;                 // log likelihood of data
        

    public:
        /** @brief Use this method to initialize a GMM fitting given data samples with the desired number of components.
         * Training is done with Expectation-Maximization algorithm.
         * @param[in] samples training set
         * @param[in] num_components desired number of clusters
         */
        GaussianMixtureModel(Eigen::MatrixXd samples, int num_components);

        /** @brief Use this method to initialize a GMM starting from mean points and covariance matrices already known.
         * Faster than fitting a GMM from a dataset. The number of component is equal to the number of elements in the input vectors.
         * @param[in] means vector of mean points, one for each cluster
         * @param[in] covariances vector of covariance matrices, one for each cluster
         * @param[in] weights mixture proportion of each cluster
         * @throw return error if size of input vectors is not consistent
         */                                       
        GaussianMixtureModel(std::vector<Eigen::VectorXd> means, std::vector<Eigen::MatrixXd> covariances, std::vector<double> weights);

        /** @brief Use this method to initialize a GMM from random mean points and covariance matrices.
         * DEPRECATED! The fitting procedure is very slow. It's better to initialize an empty GMM and then set the parameters with the setter methods.
         * @param[in] num_components number of clusters
         */  
        GaussianMixtureModel(int num_components);                                                                    // constructor: create GMM with random means and covariances    

        /** @brief Use this method to initialize an empty GMM. Parameters must be set with the setter methods.
         */  
        GaussianMixtureModel();
        
        /** @brief Destructor
         */  
        ~GaussianMixtureModel();        
        
        /** @brief Expectation-Maximization algorithm. Fits a GMM to a dataset recursively iterating between E-step and M-step.
         * Iterations stop when the log likelihood of the data does not change more than a specified tolerance or when the maximum number of iterations is reached.
         * The GMM must be already initialized.
         * @param[in] samples matrix of data samples. Each column is a sample.
         * @param[in] num_components number of clusters
         * @param[in] max_iterations maximum number of iterations (default: 1000)
         * @param[in] tolerance convergence threshold (default: 1e-3)
         * @param[in] verbose print number of iterations, final log_likelihood and time required for computation (default: false)
         */
        void fitgmm(Eigen::MatrixXd samples, int num_components, int max_iterations, double tolerance, bool verbose);                     // expectation maximization algorithm

        /** @brief Expectation-Maximization algorithm. Fits a GMM to a dataset recursively iterating between E-step and M-step.
         * Iterations stop when the log likelihood of the data does not change more than a specified tolerance or when the maximum number of iterations is reached.
         * The GMM must be already initialized.
         * @param[in] samples matrix of data samples. Each column is a sample.
         * @param[in] num_components number of clusters
         * @param[in] max_iterations maximum number of iterations (default: 1000)
         * @param[in] tolerance convergence threshold (default: 1e-3)
         * @param[in] verbose print number of iterations, final log_likelihood and time required for computation (default: false)
         */
        void fitgmm2(Eigen::MatrixXd samples, int num_components, int max_iterations, double tolerance, bool verbose);                     // expectation maximization algorithm

        /** @brief Expectation-Maximization algorithm. Fits a GMM to a dataset recursively iterating between E-step and M-step.
         * Iterations stop when the log likelihood of the data does not change more than a specified tolerance or when the maximum number of iterations is reached.
         * The GMM must be already initialized. Result is the same as above, but convergence should be faster.
         * @param[in] samples vector of samples. Each element of the std::vector contains coordinates of a sample.
         * @param[in] num_components number of clusters
         * @param[in] max_iterations maximum number of iterations (default: 1000)
         * @param[in] tolerance convergence threshold (default: 1e-3)
         * @param[in] verbose print number of iterations, final log_likelihood and time required for computation (default: false)
         */
        void fitgmm(std::vector<Eigen::VectorXd> samples, int num_components, int max_iterations, double tolerance, bool verbose);                     // expectation maximization algorithm
        
        /** @brief Mean getter
         * @return means vector of mean points, one for each cluster
        */  
        std::vector<Eigen::VectorXd> getMeans();

        /** @brief Covariances getter
         * @return covariances vector of covariance matrices, one for each cluster
        */  
        std::vector<Eigen::MatrixXd> getCovariances();

        /** @brief Weights getter
         * @return weights vector of weights, one for each cluster
        */  
        std::vector<double> getWeights();

        /** @brief Log-likelihood getter
         * @return logLikelihood log-likelihood value of the data
        */  
        double getLogLikelihood();

        /** @brief Mean points setter method.
         * @param[in] means vector of mean points of each Gaussian component
         */
        void setMeans(std::vector<Eigen::VectorXd> means);

        /** @brief Covariance matrices setter method.
         * @param[in] covariances vector of covariance matrices of each Gaussian component
         */
        void setCovariances(std::vector<Eigen::MatrixXd> covariances);

        /** @brief Weights setter method.
         * @param[in] weights vector of weights of each Gaussian component
         */
        void setWeights(std::vector<double> weights);

        /** @brief Check dimensions consistency of the GMM parameters. 
         * @return true if dimensions are consistent, false otherwise
         */
        bool check();




        /** @brief Use this method to calculate the probability of a given 2D point from a Bivariate Gaussian Distribution (single component).
         * @param[in] q considered point coordinates
         * @param[in] mean mean point of the Bivariate Gaussian Distribution
         * @param[in] cov covariance matrix of the Bivariate Gaussian Distribution
         */
        double gauss_pdf_2d(Eigen::VectorXd q, Eigen::VectorXd mean, Eigen::MatrixXd cov);

};

#endif // GAUSSIAN_MIXTURE_MODEL_H