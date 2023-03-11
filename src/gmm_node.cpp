#include <ros/ros.h>
#include <tf2/utils.h>
#include <random>

#include "gaussian_mixture_model/gaussian_mixture_model.h"



class GMM_Test
{
private:
    ros::NodeHandle n;
    ros::Timer timer;
    GaussianMixtureModel gmm;
    std::vector<Eigen::MatrixXd> covariances;
    std::vector<Eigen::VectorXd> mean_points;
    std::vector<double> weights;
    Eigen::MatrixXd samples;

public:

    // initialize GMM with 5 random components
    GMM_Test(): gmm(4)
    {
        std::cout << "Constructor called" << std::endl;
        timer = n.createTimer(ros::Duration(1.0), &GMM_Test::timerCallback,this);

        samples.resize(2,200);
        double dev = 2.0;
        std::default_random_engine gen;

        // Set desired values
        Eigen::VectorXd p1(2);
        p1 << -6.0, 0.0;
        // Generate samples
        std::normal_distribution<double> dist_x(p1(0), dev);
        std::normal_distribution<double> dist_y(p1(1), dev);

        for(int i = 0; i < 50; i++)
        {
            samples(0,i) = dist_x(gen);
            samples(1,i) = dist_y(gen);
        }

        // Set desired values
        Eigen::VectorXd p2(2);
        p1 << -4.0, 4.0;
        // Generate samples
        std::normal_distribution<double> dist_x2(p2(0), dev);
        std::normal_distribution<double> dist_y2(p2(1), dev);

        for(int i = 50; i < 100; i++)
        {
            samples(0,i) = dist_x2(gen);
            samples(1,i) = dist_y2(gen);
        }


        // Set desired values
        Eigen::VectorXd p3(2);
        p1 << 4.0, 6.0;
        // Generate samples
        std::normal_distribution<double> dist_x3(p3(0), dev);
        std::normal_distribution<double> dist_y3(p3(1), dev);

        for(int i = 100; i < 150; i++)
        {
            samples(0,i) = dist_x3(gen);
            samples(1,i) = dist_y3(gen);
        }

        // Set desired values
        Eigen::VectorXd p4(2);
        p1 << 4.0, -6.0;
        // Generate samples
        std::normal_distribution<double> dist_x4(p4(0), dev);
        std::normal_distribution<double> dist_y4(p4(1), dev);

        for(int i = 150; i < 200; i++)
        {
            samples(0,i) = dist_x4(gen);
            samples(1,i) = dist_y4(gen);
        }

        gmm.fitgmm(samples, 4, 1000, 1e-3, true);

        mean_points = gmm.getMeans();
        covariances = gmm.getCovariances();
        weights = gmm.getWeights();

        std::cout << "GMM Initialized" << std::endl;

    }

    void timerCallback(const ros::TimerEvent&)
    {
        std::cout << "Mean points: \n";
        for (int i = 0; i < mean_points.size(); i++)
        {
            std::cout << mean_points[i].transpose() << std::endl;
        } 

        std::cout << "Covariance matrices:\n";
        for (int i = 0; i < covariances.size(); i++)
        {
            std::cout << covariances[i].transpose() << std::endl;
        }

        std::cout << "Mixture proportion: ";
        for (int i = 0; i < weights.size(); i++)
        {
            std::cout << weights[i];
        }
        std::cout << std::endl;
    }

    

};//End of class SubscribeAndPublish

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gmm_test_node");
    GMM_Test gmm_node;

    ros::spin();

    return 0;
}