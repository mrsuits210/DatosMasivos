
// Inicializamos los arreglos con los scores de las temporadas

var arreglo0 = Array(10,5,20,20,4,5,2,25,1) 
var arreglo1 = Array(3,4,21,36,10,28,35,5,24,42) 

//Creamos la funcion Juego en la cual se ejecturan las posiciones del arreglo y calculara cuantas veces se rompio el record
// con la mayor cantidad de puntos y la menor cantidad de puntos. 

def Juego(score: Array[Int]): Unit = {
    var scoremax = score(0)
    var scoremin = score(0)
    // Contadores para recorrer los ciclos
    var ContadorMax = 0 
    var ContadorMin = 0 
    // Si n que es el numero de juegos es menor al contador Min se hace un incremento de +1  para recorrer El arreglo

    for(n <- score){
        if(n < scoremin){
            // Se asigna el valor para calcular el decremento. 
        scoremin = n 
        ContadorMin = ContadorMin + 1 
    }
    // Sin n es el numero de juegos es menor al contador Max se hace un incremento de +1 Para recorrer el Arreglo
    if(n > scoremax)
    {
            // se asigna el valor para calcular el Incremento 
        scoremax = n 
        ContadorMax = ContadorMax + 1 
    }

}
// ContadorMax Muestra las veces en las cuales rompio su record Mayor 
// ContadorMin Muestra las veces en las cuales rompio su record Menor. 
  println(ContadorMax,ContadorMin)

}
// Mandar Parametros de los Arreglos a la funcion 
Juego(arreglo0)
Juego(arreglo1)
