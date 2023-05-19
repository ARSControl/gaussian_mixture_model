#!/usr/bin/env python3


import rospy
import sys
from geometry_msgs.msg import Point
from gmm_msgs.msg import GMM, Gaussian


class GMMPub():
    def __init__(self):
        rospy.init_node("gmm_publisher")
        
        pub = rospy.Publisher("/gaussian_mixture_model", Point, queue_size=1)       

        self.pub = pub



def main():
    c = GMMPub()
    p = Point()
    p.x = 1
    p.y = 2
    p.z = 3
    while not rospy.is_shutdown():
        c.pub.publish(p)
        rospy.sleep(1)
    # rospy.spin()


if __name__ == '__main__':
    main()
    


