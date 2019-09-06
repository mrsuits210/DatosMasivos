// serie Fibonnaci 


// recursivo 
 def fib(n:Int) : Int = {
     if (n<2){n  }
     else { fib(n-1)+fib(n-2)}


 }
 for (i<-1 to 10)
 println(fib(i))

//explicacion 
 // Link GitHUB 

 def fibi(n:Int, arr:Int) Int , int ={

    for(int i = 0; i < n; i++){
            if(i < 2)
               arr[i] = 1;}
            else
               arr[i] = arr[i-1] + arr[i-2];
         return arr[n-1];
 }
 
println(fibi(10))



