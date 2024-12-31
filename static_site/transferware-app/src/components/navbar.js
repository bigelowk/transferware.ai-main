import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <div className=" flex flex-row items-center justify-between bg-white shadow-md w-full z-10 fixed py-3 px-6 lg:px-8 " id={"navbar"}>
      <Link to={"/"} className="text-lg md:text-2xl md:ml-12 font-semibold">
        Transferware.<span className="text-blue-500">ai</span>
      </Link>
      <ul className="flex font-semibold space-x-2 sm:space-x-8 sm:text-lg">
        <li className="hover:text-blue-500">
          <Link to="/">Home</Link>
        </li>
        <li className="hover:text-blue-500">
          <Link to="/about">About</Link>
        </li>
        <li className="hover:text-blue-500">
          <Link to="/uploadPage">Upload</Link>
        </li>
      </ul>
    </div>
  );
};

export default Navbar;
