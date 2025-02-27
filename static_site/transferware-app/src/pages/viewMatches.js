import React from "react";
import { useLocation } from "react-router-dom"; // Import useLocation
import MatchResults from "../components/matchResults";
import Footer from "../components/footer";

function ViewMatches() {
  const location = useLocation(); // Retrieve location object
  const { imagePreviewUrl } = location.state || {}; // Destructure imagePreviewUrl from state, defaulting to an empty object if state is undefined

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
