nn(nop_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: number_of_persons(X,Y).

difference(X1, X2, Y) :- number_of_persons(X1, Y1), number_of_persons(X2,Y2), Y is Y2-Y1.




