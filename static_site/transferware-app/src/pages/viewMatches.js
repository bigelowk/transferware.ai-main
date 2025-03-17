import React, { useEffect, useState  } from "react";
import { useLocation } from "react-router-dom";
import MatchResults from "../components/matchResults";
import Footer from "../components/footer";



function ViewMatches() {
  const location = useLocation();
  const { imagePreviewUrl } = location.state || {};


  const submitSurvey = () => {
    fetch("/api/analytics_id/")
      .then(response => response.json())
      .then(data => {
        const resultId = data.result_id || null;
        localStorage.setItem("result_id", resultId || "");
        console.log("Fetched result_id:", resultId);
  
        if (resultId) {
          document.location.assign(`http://transferware-ai.umd.umich.edu:5001?analytics_id=${resultId}`);
        } else {
          console.warn("Result ID not available yet");
        }
  
        // Close modal if exists
        const modal = document.getElementById("myModal");
        if (modal) modal.style.display = "none";
      })
      .catch(error => console.error("Error fetching result_id:", error));
  };

  useEffect(() => {
    const modal = document.getElementById("myModal");
    const btn = document.getElementById("surveyButton");
    const span = document.getElementsByClassName("close")[0];

    if (btn) {
      btn.onclick = function () {
        if (modal) modal.style.display = "block";
      };
    }

    if (span) {
      span.onclick = function () {
        if (modal) modal.style.display = "none";
      };
    }

    window.onclick = function (event) {
      if (modal && event.target === modal) {
        modal.style.display = "none";
      }
    };
  }, []);
  //const location = useLocation(); // Retrieve location object
  //const { imagePreviewUrl } = location.state || {}; // Destructure imagePreviewUrl from state, defaulting to an empty object if state is undefined

  return (
    <div>
      <div className="flex flex-col xl:flex-row">
        {imagePreviewUrl ? (
          <div className="flex items-start justify-center p-12 bg-zinc-900">
            <div className="flex flex-col items-center">
              <div className=" mb-4 p-6">
                <h1 className="font-semibold text-white text-2xl font-serif ">
                  Your Match Results
                </h1>
                <hr className="rounded w-1/2 h-1  my-4 border-0 rounded bg-amber-600"></hr>
                <p className="text-white font-serif">
                  These are transferware pieces from our database that look
                  similar to the sherd that you attached. The lower the
                  confidence number is, the closer of a match the pattern is.
                  Navigative to the TCC website url to get more information on
                  each pattern.
                </p>
                <hr className="rounded w-1/2 h-1 my-4 border-0 bg-amber-600" />
                <p className="text-white font-serif">
                  The confidence value is the percentage of similarity between
                  the submitted image and the resulting patterns. A percentage
                  of 100% means it is a perfect match.
                </p>
                <hr className="rounded w-1/2 h-1 my-4 border-0 bg-amber-600" />
              </div>
              <img className="max-w-80" src={imagePreviewUrl} alt="Uploaded Preview" />
              <div className="flex items-center justify-center p-4">
                <button
                  id="surveyButton"
                  onClick={submitSurvey} 
        
                  className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-700">
                  Beta Testing Survey
                </button>
              </div>
              <img
                className="max-w-80"
                src={imagePreviewUrl}
                alt="Uploaded Preview"
              />
            </div>
          </div>
        ) : (
          <p>No image preview available</p>
        )}
        
        <div className="flex1 justify-center items-center overflow-y-auto">
          <MatchResults />
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default ViewMatches;
