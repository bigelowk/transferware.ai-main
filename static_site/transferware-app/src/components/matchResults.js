import React, { useEffect } from "react";
import { useData } from "../DataContext";

const MatchResults = () => {
  const { data } = useData();

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

  const submitSurvey = () => {
    alert("Survey submitted! Thank you.");
    const modal = document.getElementById("myModal");
    if (modal) modal.style.display = "none";
  };

  if (!data) return <p>No data available</p>;

  return (
    <div className="flex items-center lg:p-4">
      <div className="flex grid grid-cols-2 lg:grid-cols-3 gap-2 p-4 w-full">
        {data.map((item) => (
          <a
            key={item.id}
            className="flex flex-col justify-center p-3 hover:border hover:shadow-sm"
            href={item.tcc_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            <div className="flex flex-row items-center lg:min-w-60 sm:p-6">
              <img
                src={item.imageUrl}
                alt="Pattern-img"
                className="w-full lg:w-full mb-2"
              />
            </div>
            <p className="mb-1 font-serif text-xl font-semibold">
              {item.pattern_name}
            </p>
            <p className="mb-4 font-light text-gray-700">
              Confidence: {item.confidence.toFixed(3)}
            </p>
          </a>
        ))}
      </div>
      <div>
        <button
          id="surveyButton"
          className="bg-black text-white font-semibold px-4 py-2 rounded-md"
          disabled={false} // Modify logic based on conditions
        >
          Survey
        </button>
      </div>

      {/* Modal */}
      <div id="myModal" className="modal">
        <div className="modal-content">
          <h1>Feedback Survey</h1>
          <label htmlFor="name">Name:</label>
          <input type="text" id="name" name="name" placeholder="Enter your name" required />

          <label htmlFor="email">Email:</label>
          <input type="text" id="email" name="email" placeholder="Enter your email" required />

          <label htmlFor="sherd-name">
            Please provide the name of the ceramic sherd you submitted:
          </label>
          <input type="text" id="sherd-name" name="sherd-name" placeholder="Sherd name" required />

          <label htmlFor="sherd-portion">Which portion of the sherd did you submit?</label>
          <select id="sherd-portion" name="sherd-portion" required>
            <option value="">Select</option>
            <option value="Border">Border</option>
            <option value="Maker's Mark">Maker's Mark</option>
            <option value="Center Design">Center Design</option>
            <option value="Other">Other</option>
          </select>
          <textarea id="other-description" name="other-description" rows="2" placeholder="If 'Other', please describe..." style={{ display: "none" }}></textarea>

          <label htmlFor="pattern-found">
            Did you find the pattern you were looking for in the results? What number is it?
          </label>
          <input type="text" id="pattern-found" name="pattern-found" placeholder="Enter pattern number" required />

          <label htmlFor="general-feedback">General Feedback:</label>
          <textarea id="general-feedback" name="general-feedback" rows="4" placeholder="Enter your feedback" required></textarea>

          <button onClick={submitSurvey}>Submit</button>
        </div>
      </div>
    </div>
  );
};

export default MatchResults;


