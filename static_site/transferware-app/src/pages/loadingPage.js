import React from "react";
import "./loadingPage.css"
import archeologistDigging from "../assets/gifs/archeologist-digging.gif"
function LoadingAnimation(){
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <div className="flex flex-col justify-center items-center ">
          <img className="h-3/5 pb-8" src={archeologistDigging}></img>
                  <div className="loader"></div>
        </div>
        </div>
    );
}
export default LoadingAnimation;
