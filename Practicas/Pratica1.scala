// Assessment 1/Practica 1
//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
val cir = 15
val pi = 2*3.1416
val rad = cir / pi


println(rad)





//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
val numero = 3
var cont = 0
for(i <- Range(1, numero + 1)) {
 if( num % i == 0) {
   cont += 1
 }
}
if(cont != 2) {
 println("Numero no primo")
} else {
 println("El numero es primo ")
}

//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"
val bird = "tweet"
val interpolar = "Estoy escribiendo un "+ bird



//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"
  val star = "Hola Luke Yo soy tu padre"
    star.slice(5,9)

//5. Cual es la diferencia en value y una variable en scala?
// La Variable Puede Cambiar de Valor y se denomina variable Mutable
// Value es una variable que no  puede cambiar  de valor y se llama variable inmutable.

//6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23))) regresa el numero 3.1416

val tupla = ((2,4,5),(1,2,3),(3.1416,23))
println(tupla._3)
