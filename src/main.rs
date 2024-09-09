use std::fmt::{Debug  };
use std::convert::TryFrom;

// use std::simd::usizex32;
use std::{iter, usize};
use std::iter::{zip}; 

use faer::reborrow::IntoConst;
// use faer::reborrow::IntoConst;
use faer::{mat , Mat };
use faer_ext::*;

use ndarray::{array, concatenate, Axis , ArrayBase , RawData , CowRepr , OwnedRepr };
use ndarray::Order;
use ndarray::prelude::*;

use rand;
use std::any::type_name;
use std::time::Instant;
use std::mem::size_of;


fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}


type Ndarray1 = ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>;
type Ndarray2 = ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>;
// <S1 : ndarray::Data  , D > 
// fn take_n_fiber(nd_tensor  : &ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 3] >> ,
//     n : usize ){

//         let shape = nd_tensor.shape();
//         let mut result  = Vec::new();
//         // nd_tensor[[i , j , k ]] ;

//     }
fn matricilyze_tensor( nd_tensor  : &ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 3] >> ,
    n : usize )
    -> ArrayBase< ndarray::OwnedRepr<f64> , Dim<[usize; 2] >> 
    // where  S1: RawData<Elem = f64>  ,  D  : Dimension<Smaller = ndarray::Dim<[usize; 2]>>  + ndarray::RemoveAxis ,
{   
    println!("A entrada foi : \n{:?}", &nd_tensor);

    let mut result : Vec<ArrayBase<CowRepr<f64>, Dim<[usize; 2]>>>  = Vec::new(); 
    let mut result2: Vec<ArrayBase<CowRepr<f64>, Dim<[usize; 2]>>>  = Vec::new();

    //++++++++++++++++ GAMBIARRA PARA CONSERTAR O FUNCIONAMENTO DO INDEX_AXIS QUE SÓ FUNCIONA CORRETAMENTE EM N == 2 :+++++++++
    let mut perm_order = [0 , 1 , 2 ];
    let mut for_range: i32 = nd_tensor.shape()[n].try_into().unwrap() ;

    let mut vet_len = &0 ;
    match n{
        0 => {  perm_order = [ 0 , 2 , 1 ] ;
                for_range = nd_tensor.shape()[n].try_into().unwrap() ;
             } ,
        1 => {  perm_order = [ 1 , 0 , 2 ];
                for_range = nd_tensor.shape()[0].try_into().unwrap() ;
             } ,
        2 => {  perm_order = [ 1 , 0 , 2 ] ;
                for_range = nd_tensor.shape()[n].try_into().unwrap() ;
             } ,
        _ => panic!("Valor Inválido de n usado na tentativa de Matricializar o Tensor de entrada "),
    }
    
    let nd_tensor = nd_tensor.clone().permuted_axes(perm_order) ;
    let mut out: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>>   = nd_tensor.index_axis(Axis(0), 0);
    // let n = 0 ;
    // println!("A entrada Permutada ficou : \n{:?}", &nd_tensor.index_axis(Axis(1), 0));
    
    for index in 0..usize::try_from(for_range).unwrap() {

        out  = nd_tensor.index_axis(Axis(n), index);
        // println!("O valor de OUT Fora do IF :\n{:?}",&out) ;
        // vet_len = &out.shape()[1] ;
        out = out.permuted_axes([1 , 0]);

        // println!("Teste de out :\n{:?}" , &out);
        // if n == 1{
        //     println!("O valor de OUT Dentro do IF :\n{:?}",&out) ;
        //     result.push( out.into() );
        // }else{

        //     result.push( out.into() );
        // }
        result.push( out.into() );
        // let mut out1 = out.rows().into_iter().collect::<Vec<_>>();
        // let mut aux = Vec::new();
        
        
        

        // for it in 0..out1.len() {
                                        
        //     // println!("{:?}", &out1[it].to_shape(((vet_len, 1), Order::RowMajor)).unwrap() );
        //     aux.push( out1[it].to_shape((( (*vet_len).into() , 1), Order::RowMajor)).unwrap() ) ;
        // }
        // println!("Resultado dos to_shape :\n{:?}", &aux );
        
        // let out1 = aux ;
        // let mut aux: Vec<ArrayBase<CowRepr<'_, f64>, Dim<[usize; 2]>>> = Vec::new();


        // for mut vetor in out1 {

        //     println!("Eu estive aki");
        //     aux.push( vetor.mapv_into(|x|convert_f64(x)));//.collect::<Vec<_>>() )  ;
        //     // vetor.into_iter().map(|x|println!("O valor é :\n{:?}",x) )   ;

        // }
        // println!("O Atual valor :\n{:?}" , &aux );

        // result.extend(aux);
        // let out1 = aux ;
        // // let mut aux = Vec::new();
        // Array2::<f64>::
        // for it in 0..out1.len(){

        //     println!("O valor é : \n{:?}" , Array2(&vec![out1[it].clone()])  );
        // }

        // out2 = aux ;
        
        
        // let mut out = out.map(|x|convert_f64(*x));
        // println!("O axis Extraído foi :\n{:?}" , &out1 );//slice_mut(s![..,..]) );// .slice_axis_inplace(Axis(0) ,  ndarray::Slice::from(..) ) );

        // for vector in out.s {

        // }

        // let out1 = out.to_shape(((out[0].shape()[0] , 1), Order::RowMajor)).unwrap(); //
        // let
        
        // println!("O Result :\n{:?}" , result ) ;
        // let out: ArrayBase<ndarray::OwnedRepr< f64 >, Dim<[usize; 2]>>   = out1.map(|x|convert_f64(*x)) ;
        // let out = out1 ;

        // result.extend(vec![ out ]);
        // result.extend( out2 );
             
    }
    // println!("O Result matricializado Pré-Concatenate com n na direção {n} foi :\n{:?}" , &result );

    let result: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = concatenate(Axis(1) , &result.iter().map(|vtor |{vtor.view()}).collect::<Vec<_>>() ).unwrap() ;
    
    // result
    println!("O Result matricializado foi :\n{:?}" , result );
    // nd_tensor.clone()
    result
    
}

fn parafac_decomposition(   nd_tensor  : ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 3] >> ,
                            r : usize ,
                            max_number_of_iterations : i32 )
                                //:::: O OUTPUT SAO 3 MATRIZES :::
    -> ( ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 2] >> , ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 2] >> , ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 2] >>  )
{   
    let n = 3 ;
    let ts_shape = nd_tensor.shape();
    //------------------ INSTANCIA N MATRIZES ALEATORIAMENTE (Onde n == ao número de n-dimensões do Tensor ): ------------------------------
    let mut r_list = vec![Array2::<f64>::zeros(( ts_shape[1] , r )) ,
                                                             Array2::<f64>::zeros(( ts_shape[2] , r )) ,
                                                             Array2::<f64>::zeros(( ts_shape[0] , r )) ];
    
    for it in 0..r_list.len(){
        println!("\nO Shape da matriz referente ao n = {it} é igual a {:?}\n" , &r_list[it].shape() );
    }
    
    for  r in &mut r_list {

        for x in r.iter_mut(){ 
            *x = rand::random();
            
        }

    }
    println!("\nIniciando o Looping de Decomposição Parafac. ..  ...\n\n");
    for iteration in 0..max_number_of_iterations {

        let mut v     = Array2::<f64>::ones(( r , r )) ;
        let mut v_aux = Array2::<f64>::zeros(( r , r )) ;
        for n in 0..r_list.len() 
        {   
            
            v     = Array2::<f64>::ones( ( r , r )) ;
            v_aux = Array2::<f64>::zeros(( r , r )) ;
            //--+-=+-+-+-+-=+--------TRECHO DO PRODUTO HADAMANT-------------------+++++++++++++
            for i in 0..r_list.len() {
                if n != i {
                    let i_perm = r_list[i].clone().permuted_axes([1, 0]) ;
                    v_aux = i_perm.dot( &r_list[i] ) ;
                    v = v*v_aux;
                }                
                
            }
            println!("A Inversa será calculada a seguir : ");
            v = nd_pseudo_inverse(v);

            println!("A Inversa foi calculada : ");
            //+++++++++++++++++------TRECHO DO PRODUTO kATRI-RAO---------------++++++++++++++++++++++++++
            // println!("Produto Khatri-Rao do n = {:?} com o n = {:?}" , r_list.len() - 1 , r_list.len() - 2 );
            let mut a_kt = r_list[r_list.len() - 1].clone() ;
            
            // if n == r_list.len() - 1{
            //     println!("Produto Khatri-Rao do n = {:?} com o n = {:?}" , r_list.len() - 1 , r_list.len() - 2 );
            //     a_kt = r_list[r_list.len() - 2].clone()  ;
            // }else if n == r_list.len() - 2 {
            //     a_kt = r_list[r_list.len() - 1].clone()  ;
            // }else{
            //     a_kt = product_khatri_rao(r_list[r_list.len() - 1 ].clone() , &r_list[r_list.len() - 2] ) ;
            // }
            //  //Array2::<f64>::zeros(( r , r )) ;
            // for i in (0..r_list.len()-3).rev()  {
            //     // in_ts{}
            //     if i != n {
            //         println!("Produto Khatri-Rao do acumulado com o n = {:?} " , i - 1 );
            //         a_kt = product_khatri_rao(a_kt.clone() , &r_list[i-1] ) ;
            //     }
                
            // }

            match n {
                0 => a_kt = product_khatri_rao(r_list[1].clone() , &r_list[2] ) ,
                1 => a_kt = product_khatri_rao(r_list[0].clone() , &r_list[2] ) ,
                2 => a_kt = product_khatri_rao(r_list[0].clone() , &r_list[1] ) ,
                _ => panic!("Valor Inválido de n usado na tentativa de Realizar o Khatri-Rao o Tensor de entrada "),
            }

            // r_list.reverse() ;
            println!("O shape Final dos Produtos Katri-Rao : {:?}", &a_kt.shape() );
            let als_aux = a_kt.dot(&v);
            // println!("Vou matricializar um tensor ");
            let Xn = matricilyze_tensor(&nd_tensor , n ) ;
            println!("\n\nTensor X matricializado com Sucesso em relação a direção n : {n}\n\n");
            // let als_aux: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = 
            // *n = ??????
            r_list[n] = Xn.dot( &als_aux );

        }


    }
    


    
    // println!("O Resultado do rand foi : \n{:?}", &r_list[0].clone().permuted_axes([1, 0]) );
    ( r_list[0].clone() , r_list[1].clone() , r_list[2].clone() )
}

fn nd_pseudo_inverse ( nd_tensor  : ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 2] >>  ) 
                                                -> ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 2] >> 
                                                
{
    let faer_tensor = ndarray_2_faer(nd_tensor) ;
    // println!("Meus valores foram convertidos ");
    let svd = faer_tensor.svd() ;
    // println!("Svd foi Calculado ! ");
    let pseudo = svd.pseudoinverse();

    
    let nd_tensor = faer_2_ndarray(pseudo.clone());
    
        

    nd_tensor
}


fn ndarray_2_faer( nd_tensor  : ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> )
-> Mat<f64> 
// where S1: RawData<Elem = f64>  , D : Dimension , 
{
    
    // println!("O nd-array de Entrada foi :\n{:?}" , &nd_tensor );

    let rows    = nd_tensor.shape()[0];
    let columns = nd_tensor.shape()[1];

    let mut I_faer = Mat::<f64>::identity(rows, columns);
    
    for i in 0..rows {
        // println!("Linha :{i}");
        for j in 0..columns {
            // println!("Coluna {j}");
            I_faer[[i,j].into()] = nd_tensor[[i , j]] ;
        }
    }


    // println!("O faer de Saída foi :\n{:?}" , &I_faer );

    I_faer
}

fn faer_2_ndarray( faer_tensor  : Mat< f64 > )
        -> ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>
        // where S1: RawData<Elem = f64>  , D : Dimension , 
    {
        // println!("O faer_tensor de Entrada foi :\n{:?}" , &faer_tensor );


        let rows: usize    = faer_tensor.shape().0 ;
        let columns: usize = faer_tensor.shape().1;


        let mut nd_tensor = Array2::<f64>::zeros((rows, columns));
        
        // let mut I_faer = Mat::<f64>::identity(rows, columns);
        
        for i in 0..rows {
            // println!("Linha :{i}");
            for j in 0..columns {
                // println!("Coluna {j}");
                // I_faer[[i,j].into()] = nd_tensor[[i , j]] ;
                nd_tensor[[i , j]] = faer_tensor[[i,j].into()] ; 
            }
        }


        // println!("O nd_tensor de Saída foi :\n{:?}" , &nd_tensor );

        nd_tensor
    }


struct in_ts<T1:std::iter::Iterator >
    {
    ts : T1 ,
}
//+ std::fmt::Debug
// Produto Tendorial de Kronecker Com saída em 1 dimensão :
fn product_kronecker_1d<T1:std::iter::Iterator  , S1 : ndarray::Data , D >( a_ts  : in_ts<T1> , 
                                                                            b_ts  : &ArrayBase<S1, D> ) 
    -> ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>> 
    where <T1 as Iterator>:: Item : Debug + Into<f64> + Clone   , S1: RawData<Elem = f64>  , D : Dimension ,  
    {
        let it = a_ts.ts;
        let mut result  = Vec::new();


        for i in it {              
            
            result.extend(vec![ i.into()*b_ts.flatten() ]);
            
        };
        let result = concatenate(Axis(0) , &result.iter().map(|vtor|{vtor.view()}).collect::<Vec<_>>() ).unwrap() ; 
        // let result = result
        // println!("Executado    , O resultado Final Foi :\n{:?}" ,  result  ); // .apply(.view())
        // println!("b_ts é :\n{:?}" , type_of( b_ts.flatten() ) );
        
        
        return result ; 
    }

fn convert_usize<T1> (value: T1 )-> usize
where T1 : IntoConst, usize: From<T1> //Into<usize>,
{

    value.into()
    // value.

}
    
fn convert_f64<T1> (value: T1 )-> f64
where T1 : Into<f64>,
{

    value.into()

}

fn product_khatri_rao(     a_ts :  ArrayBase<ndarray::OwnedRepr< f64 > , ndarray::prelude::Dim<[usize; 2]>> , 
                           b_ts : &ArrayBase<ndarray::OwnedRepr< f64 > , ndarray::prelude::Dim<[usize; 2]>> ) 
                           ->      ArrayBase<ndarray::OwnedRepr< f64 > , ndarray::prelude::Dim<[usize; 2]>> 
                           
                       {
    
    let mut result: Vec<        ArrayBase<ndarray::OwnedRepr< f64 > , ndarray::prelude::Dim<[usize; 2]>> >  = Vec::new();
    // let out : ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>  =  array![4.0 , 2.0 ];

    for (a_j , b_j) in zip(a_ts.columns() , b_ts.columns()  ){ //columns_mut()// .axis_iter_mut(Axis(0))
            //a_j.mapv_into(|vl|{vl*40.0})
            
            let inp = in_ts{ts : a_j.into_iter().map(|&x| convert_f64(x) ) } ;
            let b_size = b_j.shape()[0];
            let inp_b = b_j.to_shape((( b_size , 1), Order::RowMajor)).unwrap() ;
            
            // println!("Coluna de A :\n{:?}" , &a_j ) ;  
            
            // println!("Colunas de B :\n{:?}\n" , &b_j );

            // println!("O Produto Tensorial entre a Coluna A e Coluna B é : \n{:?}" , product_Kronecker_1d( inp , &inp_b ) );
            
            let inp = in_ts{ts : a_j.into_iter().map(|&x| convert_f64(x) ) } ;
            // result.push(vec![ product_Kronecker_1d( inp , &inp_b ).to_shape(((4, 1), Order::RowMajor)).unwrap()  ]);
            let out: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>   = product_kronecker_1d( inp , &inp_b ) ;
            let out_size = out.shape()[0] ;
            let out1 = out.to_shape((( out_size , 1), Order::RowMajor)).unwrap(); //
            let out: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>   = out1.map(|x|convert_f64(*x)) ;
            // let out = out.collect::<Vec<_>>() ;

            // println!("O Resultado atual do out do Katri-Rao : \n{:?}" , &out );
            result.extend(vec![out]);

        
    }
    // println!("O result atual é : \n{:?}" ,  &result  );
    let result = concatenate(Axis(1) , &result.iter().map(|vtor|{vtor.view()}).collect::<Vec<_>>() ).unwrap() ;
    
    println!("Teste do Resultado final do Produto Khatri-Rao tem o shape : \n{:?}" , &result.shape() );
    
    result
}

fn product_hadamard( a_ts : &ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>> , 
    b_ts : &ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>> )-> ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>> {
    a_ts*b_ts 
}// O Produto Point-Wise dos tensores A e B



fn main() {
    
    
    // let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    // let t = t * 2;
    // let t1 = t.copy() + 2;
    // t.print();
    // println!("Isso é outro tensor {:?}" , t1 );

    // let a: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>    = array![
    //     [1.0, 2.0],
    //     [3.0, 4.0]
    //     ];
    // let a = array![
    //     [-3.0  , -5.0 ],
    //     [-12.0 , -13.0]];

    // let b = array![
    //     [1.0, 2.0],
    //     [3.0, 4.0]
    //     ];
    // let a2 = array![
    //     [[3.0 , 5.0],
    //     [12.0 , 13.0]] ,

    //     [[-3.0 , -5.0],
    //     [-12.0 , -13.0]]
    // ];
    // let teste = product_hadamard(&a,&b) ; 
    // let c = &a*&b;
    // let d = &a.dot(&b) ;
    
    

    // let a3 = in_ts {ts : a2.into_iter()};// .iter()};
    // product_Kronecker_1d(a3 , &b);

    // let ts = product_khatri_rao( b.clone()  , &b);

    // println!("O Resultado do Produto Katri-Rao é igual a :\n\n{:?}" , ts[[1 , 0]] );

    // let mut I_faer = Mat::<f64>::identity(4, 2);
    // I_faer[[0,0].into()] = ts[[1 , 0]] ;

    // println!("O Faer não alterado : \n{:?}" , I_faer[[0,0].into()] );

    
    // let faer_new = ndarray_2_faer(ts);
    
    // let x = mat![
    // [10.0, 3.0],
    // [2.0, -10.0],
    // [3.0, -45.0],
    // ];

    // let nd_new = faer_2_ndarray(x.clone());

    // let svd = x.svd();
    // let ps = svd.pseudoinverse();
    // println!("svd : \n{:?}" , svd );
    // println!("ps : \n{:?}" , ps );

    // let nd_pseudo = nd_pseudo_inverse(nd_new);
    

    // println!("Ultima matrix :\n{:?}" , nd_pseudo );

    // println!("a.dot(b) = :\n{:?}\n\nb.dot(&a) :\n{:?}" , a.dot(&b) , b.dot(&a) );
    
    //++++++++CÓDIGO RESPONSÁVEL POR CONTROLAR O TAMANHO DA MEMÓRIA STACK ASSOCIADA A ESSE PROGRAMA +++++++++++++
    const N: usize = 1_000_000;
    
    std::thread::Builder::new()
        .stack_size(size_of::<f64>() * N)
        .spawn(||{
    
    let nd_teste = array![ [[1.0  , 2.0 , 3.0  ]   ,
                                                          [2.0  , 4.0 , 5.0  ] ] ,

                                                         [[-10.0 ,-25.0,36.0 ]   ,
                                                          [ -29.0,43.0 ,55.0 ]]
                                                                                ];
    // let mt_ts = matricilyze_tensor(  &nd_teste , 1 );

    // println!("O tensor a matricializado : \n{:?}" , mt_ts);

    let before = Instant::now();
    parafac_decomposition(nd_teste , 6 , 500 );
    let after = Instant::now();
    println!("\nTime elapsed: {:?}", after.duration_since(before));
    // println!("{:?} ", &a);
    
    }).unwrap().join().unwrap();
//++++++++++++++++++++++++++++++++++++++++::::BANIDO:::+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



}




// LIBTORCH
// C:\Users\limaa\libtorch

// $Env:LIBTORCH = "C:\Users\limaa\libtorch"
// $Env:Path += ";C:\Users\limaa\libtorch\lib"