use std::fmt::{Debug  };

use std::{iter, usize};
use std::iter::{zip}; 

use faer::reborrow::IntoConst;
use faer::{mat , Mat };
use faer_ext::*;

use ndarray::{array, concatenate, Axis , ArrayBase , RawData , CowRepr };
use ndarray::Order;
use ndarray::prelude::*;

// extern crate nalgebra as na;
// use nalgebra::{SVD , dmatrix } ;
// use nalgebra::*;

// use nalgebra_lapack::SVD;


use std::any::type_name;


fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}


// type ndarray1 = ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>;
// type ndarray2 = ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>;

fn nd_pseudo_inverse ( nd_tensor  : ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 2] >>  ) 
                                                -> ArrayBase< ndarray::OwnedRepr<f64> , Dim< [usize; 2] >> 
                                                
{
    let faer_tensor = ndarray_2_faer(nd_tensor) ;
    let svd = faer_tensor.svd() ;
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

fn faer_2_ndarray( faer_tensor  : Mat<f64> )
        -> ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>
        // where S1: RawData<Elem = f64>  , D : Dimension , 
    {
        println!("O faer_tensor de Entrada foi :\n{:?}" , &faer_tensor );


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


        println!("O nd_tensor de Saída foi :\n{:?}" , &nd_tensor );

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

fn convert_f64<T1> (value: T1 )-> f64
where T1 : Into<f64>,
{

    value.into()

}

fn product_khatri_rao(  a_ts :  ArrayBase<ndarray::OwnedRepr< f64 > , ndarray::prelude::Dim<[usize; 2]>> , 
                           b_ts : &ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>> ) 
                           ->      ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>> 
                           
                       {
    
    let mut result: Vec< ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>> >  = Vec::new();
    // let out : ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>  =  array![4.0 , 2.0 ];

    for (a_j , b_j) in zip(a_ts.columns() , b_ts.columns()  ){ //columns_mut()// .axis_iter_mut(Axis(0))
            //a_j.mapv_into(|vl|{vl*40.0})
            
            let inp = in_ts{ts : a_j.into_iter().map(|&x| convert_f64(x) ) } ;

            let inp_b = b_j.to_shape(((2, 1), Order::RowMajor)).unwrap() ;
            
            // println!("Coluna de A :\n{:?}" , type_of(a_j) ) ;  
            
            // println!("Colunas de B :\n{:?}\n" , &inp_b );

            // println!("O Produto Tensorial entre a Coluna A e Coluna B é : \n{:?}" , product_Kronecker_1d( inp , &inp_b ) );
            
            let inp = in_ts{ts : a_j.into_iter().map(|&x| convert_f64(x) ) } ;
            // result.push(vec![ product_Kronecker_1d( inp , &inp_b ).to_shape(((4, 1), Order::RowMajor)).unwrap()  ]);
            let out: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>   = product_kronecker_1d( inp , &inp_b ) ;
            let out1 = out.to_shape(((4, 1), Order::RowMajor)).unwrap(); //
            let out: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>   = out1.map(|x|convert_f64(*x)) ;
            // let out = out.collect::<Vec<_>>() ;

            // println!("O Resultado atual do out do Katri-Rao : \n{:?}" , &out );
            result.extend(vec![out]);

        
    }
    // println!("O result atual é : \n{:?}" ,  &result  );
    let result = concatenate(Axis(1) , &result.iter().map(|vtor|{vtor.view()}).collect::<Vec<_>>() ).unwrap() ;
    
    println!("Teste do Resultado final do Produto Katri-Rao : \n{:?}" , &result );
    
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
    let b = array![
        [1.0, 2.0],
        [3.0, 4.0]
        ];
    let a2 = array![
        [[3.0 , 5.0],
        [12.0 , 13.0]] ,

        [[-3.0 , -5.0],
        [-12.0 , -13.0]]
    ];
    // let teste = product_hadamard(&a,&b) ; 
    // let c = &a*&b;
    // let d = &a.dot(&b) ;
    
    

    let a3 = in_ts {ts : a2.into_iter()};// .iter()};
    // product_Kronecker_1d(a3 , &b);

    let ts = product_khatri_rao( b.clone()  , &b);

    // println!("O Resultado do Produto Katri-Rao é igual a :\n\n{:?}" , ts[[1 , 0]] );

    // let mut I_faer = Mat::<f64>::identity(4, 2);
    // I_faer[[0,0].into()] = ts[[1 , 0]] ;

    // println!("O Faer não alterado : \n{:?}" , I_faer[[0,0].into()] );

    
    let faer_new = ndarray_2_faer(ts);
    
    let x = mat![
    [10.0, 3.0],
    [2.0, -10.0],
    [3.0, -45.0],
    ];

    let nd_new = faer_2_ndarray(x.clone());

    let svd = x.svd();
    let ps = svd.pseudoinverse();
    // println!("svd : \n{:?}" , svd );
    // println!("ps : \n{:?}" , ps );

    let nd_pseudo = nd_pseudo_inverse(nd_new);
    

    println!("Ultima matrix :\n{:?}" , nd_pseudo );


//++++++++++++++++++++++++++++++++++++++++::::BANIDO:::+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // let matrix_teste = na::dmatrix![[12.0 , 35.0 , 27.0],[21.0, 25.0 , 98.0] , [22.0 , 23.0 , 27.0]];
    // // let (u , v , d) : () ;
    // let deco = SVD::new(matrix_teste); 


    // let x = na::dmatrix![[12.0 , 35.0 , 27.0],[21.0, 25.0 , 98.0] , [22.0 , 23.0 , 27.0]];
    // x = x.into() ;

    // let svd_v = SVD::new(x.clone().into() , true , true ).v_t.unwrap();
    // let svd_u = SVD::new(x.clone() , true , true ).u.unwrap()  ;

    // println!("svd_v = \n{:?}" , svd_v );
    // let x_pseudo = x.pseudo_inverse(0.0000001).unwrap();


    // SVD(matrix_teste);
    // ndarray_linalg.
    // deco.singular_values;  
    // println!("Teste do ndalgebra : \n{:?}" , matrix_teste); 
    // println!("a:\n{:?}\n\nm2:\n{:?}", a.what_is_this , m2.what_is_this);

}




// LIBTORCH
// C:\Users\limaa\libtorch

// $Env:LIBTORCH = "C:\Users\limaa\libtorch"
// $Env:Path += ";C:\Users\limaa\libtorch\lib"