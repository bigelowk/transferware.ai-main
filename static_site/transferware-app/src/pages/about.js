import React from "react";
import "./loadingPage.css";
import underline1 from "../assets/images/zigzag-underline.png";
import plate1 from "../assets/images/plate1.jpg";
import plate2 from "../assets/images/plate2.jpg";
import plate3 from "../assets/images/plate3.png";
import Footer from "../components/footer";
import lightIcon from "../assets/images/Light-On.png";
import focusIcon from "../assets/images/focus-icon.png";
import cleanIcon from "../assets/images/sheet-icon.png";

const Section = ({ title, text, imgSrc, reverse }) => (
  <div
    className={`w-screen lg:px-24 py-6 flex flex-row items-center justify-center ${
      reverse ? "flex-row-reverse" : ""
    } md:space-x-14`}
  >
    <img src={imgSrc} className="lg:w-1/6 md:w-1/4 w-1/2 sm:px-4 px-5" />
    <div className="flex flex-col w-1/3">
      <h1 className="font-bold text-1xl">{title}</h1>
      <p className="font-light">{text}</p>
    </div>
  </div>
);

const BestPractice = ({ icon, title, text }) => (
  <div className="flex flex-col items-center md:w-1/5 w-1/4">
    <img src={icon} className="sm:w-1/6 w-1/4" />
    <h1 className="font-bold text-1xl">{title}</h1>
    <p className="font-light text-center">{text}</p>
  </div>
);

const Underline = () => <img src={underline1} className="lg:w-2/12 w-1/3" />;

function AboutPage() {
  return (
    <div className="flex flex-col items-center pt-[52px]">
      <div className="bg-custom-image bg-cover bg-center h-64 w-screen shadow-md relative">
        <div className="absolute inset-0 flex items-center justify-center">
          {/* <button className="text-white text-4xl font-bold bg-black rounded-lg px-6">About us</button> */}
        </div>
      </div>
      <div>
        <div className="flex flex-col items-center">
          <h1 className="lg:text-2xl sm:text-xl font-semibold pt-9 pb-3">
            Find a Match to Your Sherds
          </h1>
          <Underline />
        </div>

        <Section
          title="Simplify Your Research"
          text="With Transferware.ai, you can effortlessly identify and date transferware sherds by matching your images to a comprehensive database of patterns. Our goal is to make your research faster and more efficient, allowing you to focus on your discoveries instead of manual identification."
          imgSrc={plate1}
        />

        <Section
          title="How It Works"
          text="Transferware.ai streamlines your workflow by allowing you to upload an image of your sherd through drag-and-drop, file browsing, or URL paste. Upon submission, the system employs machine learning technology to process the image and find the nearest matches."
          imgSrc={plate2}
          reverse
        />

        <Section
          title="Matching and Results"
          text="Transferware.ai delivers comprehensive results, including pattern names, confidence values, and direct links to the Transferware Collectors Club (TCC) database for further reference and detailed information."
          imgSrc={plate3}
        />

        <div className="lg:px-24 flex flex-col items-center mb-16">
          <h1 className="lg:text-2xl sm:text-xl font-semibold pb-3">
            Best Practices for Uploading Images
          </h1>
          <Underline />
          <div className="flex flex-row justify-center mt-6 space-x-3">
            <BestPractice
              icon={lightIcon}
              title="Lighting"
              text="Ensure the image is well-lit and free from shadows and glares. Natural light is ideal. "
            />
            <BestPractice
              icon={focusIcon}
              title="Focus"
              text="The sherd should be in clear focus with no blurriness. Focusing on the center pattern of the sherd often yields better results."
            />
            <BestPractice
              icon={cleanIcon}
              title="Background"
              text="Place the sherd on a plain, contrasting background to avoid distractions and errors."
            />
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default AboutPage;
