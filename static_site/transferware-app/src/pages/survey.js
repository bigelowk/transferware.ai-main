import React from "react";
import { Link } from "react-router-dom";
import group3 from "../assets/images/Group-3.png";
import uploadIcon from "../assets/images/upload-icon.png";
import viewIcon from "../assets/images/view-icon.png";
import databaseIcon from "../assets/images/database-icon.png";
import Footer from "../components/footer";

function Landing() {
  return (
    <div className="h-screen relative">
      <div className="flex flex-col lg:flex-row">
        <div className="mt-20 header-column flex flex-col justify-center pb-0 pt-10 px-0 lg:py-16 sm:px-20">
          <div className="header lg:w-full px-10 sm:px-0">
            <h1 className="font-bold text-3xl lg:text-6xl my-6">
              Upload your sherd, Find your pattern
            </h1>
            <p className="my-8 lg:w-4/5">
              Welcome to transferware.ai! Our tool was designed to be used by
              anyone, not just archaeologists. Simply upload an image of your
              sherd or plate and our model will provide the top matches for you.
            </p>
            <Link to="/uploadPage">
              <button className="bg-black font-semibold text-white p-4 px-14 rounded-lg rounded-bl-none">
                Use tool
              </button>
            </Link>
          </div>
          <div className="flex flex-row space-x-5 sm:space-x-10 my-20 font-semibold text-xs px-8 sm:px-0">
            <span className="flex flex-row items-center">
              <img
                src={uploadIcon}
                className="h-[20px] px-5"
                alt="upload-icon"
              />
              Upload sherd
            </span>
            <span className="flex flex-row items-center">
              <img src={viewIcon} className="h-[20px] px-5" alt="view-icon" />
              view 10 closest matches
            </span>
            <span className="flex flex-row items-center">
              <img
                src={databaseIcon}
                className="h-[20px] px-5"
                alt="database-icon"
              />
              get info from our database
            </span>
          </div>
        </div>
        <div className="flex flex-col justify-end items-end w-full lg:h-screen">
          <img src={group3} className="w-full" />
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default Landing;
