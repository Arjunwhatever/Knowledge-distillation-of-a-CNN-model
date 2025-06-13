Created a distilled version of a CNN I had made earlier (that was trained on CIFAR 10 datasets)

A few changes I've noted when both the Teacher and Student models are compared 


TEACHER had an accuracy of 74.84% while 
STUDENT has  an accuracy of 73.06%


TEACHER-
Conv channels = 3 to 32 to 64
Fully Connected Hidden Units = 128
Total Parameters = 1.86M
Model Size = 7.4 MB
Pooling Strategy = Conservative (stride=1 then 2)


STUDENT-
Conv channels = 3 to 16 to 32
Fully Connected Hidden Units = 64
Total Parameters = 134K
Model Size = .55MB
Pooling Strategy =  Aggressive (stride=2, 2) 

PS. I've also added a html file for visualizing the change 
