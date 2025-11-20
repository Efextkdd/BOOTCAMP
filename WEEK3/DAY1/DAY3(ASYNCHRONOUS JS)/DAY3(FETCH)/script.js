// Arrow functions
()=>{}

const greeting = () => {
    console.log('Hello world');
};
greeting();
 
// implict return 

// regular anonymous function 
const add = function(a,b){
    return a + b;
};

// converted to an arrow function
const adds = (a,b) => a+b;


const img = document.querySelector('img')
const button = document.querySelector('button');

function fetchDogImage(){
    fetch('https://dog.ceo/api/breeds/image/random')
    .then(Response => Response.json())
    .then(data => img.src = data.message) // access objects in JSON file
    // .then(data => console.log(data))
    
    .catch(error => {
        console.error('Something went wrong', error);
        img.alt = 'Failed to fetch dog image';

    }
    );
    
    

}
// fetchDogImage()

button.addEventListener('click', fetchDogImage);

// async / await

// async function fetchDogImage() {
//     const response = await fetch('https://dog.ceo/api/breeds/image/random')
//     const jsonData = await response.json();
//     img.src = jsonData.message;
// }
// button.addEventListener('click', fetchDogImage);