// 1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"
 val lista = List("rojo","blanco","negro")
 println(lista)

// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
val c1 = "Verde" :: lista
val c2 = "Amarillo" :: c1
val c3 = "Azul" :: c2
val c4 = "Naranja" :: c3
val c5 = "Perla" :: c4

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"

var lista = List("rojo", "blanco", "negro","verde","amarillo","azul", "naranja", "perla")
lista.slice(3,6)
// 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5

val array = (1 to 1000).by(5)
    for(i <- array){
        println(""+i)

    }

// 5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos
  val mylist = List(1,3,3,4,5,7,3,7)
    mylist.toSet
// 6. Crea una mapa mutable llamado nombres que contenga los siguiente
//     "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
//  Imprime todas la llaves del mapa
val nombres = collection.mutable.Map(("Jose",20),("Luis",24),("Ana",23),("Susana",27))
nombres.keys
