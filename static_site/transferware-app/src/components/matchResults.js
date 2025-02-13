import React from "react";
import { useData } from "../DataContext";

var modal = document.getElementById("myModal");
    
// Get the button that opens the modal
var btn = document.getElementById("surveyButton");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks the button, open the modal 
btn.onclick = function() {
  modal.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
  modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}

// Function to handle survey submission
function submitSurvey() {
  alert("Survey submitted! Thank you.");
  modal.style.display = "none"; // Close the modal after submission
}

// Example to enable the Survey button when there is a selected file
// You can replace this logic with your own implementation.
const selectedFile = "example.txt"; // Example file to enable the button
if (selectedFile) {
  btn.disabled = false;
}

const MatchResults = () => {
  const { data } = useData();

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
          >
            <div className="flex flex-row items-center lg:min-w-60 sm:p-6">
              <img
                src={item.imageUrl}
                alt="Pattern-img"
                className="w-full lg:w-full mb-2"
              />
            </div>
            <p className="mb-1 font-serif text-xl font-semibold">
              <span className="font-semibold "></span> {item.pattern_name}
            </p>
            <p className="mb-4 font-light text-gray-700">
              <span className="">Confidence:</span> {item.confidence.toFixed(3)}
            </p>
          </a>
        ))}
      </div>
      <div> 
        <button
          id="surveyButton"
          class="bg-black text-white font-semibold px-4 py-2 rounded-md"
          disabled
        >
          Survey
        </button>
        
      </div>
    
      <div id="myModal" class="modal">
        <div class="modal-content">
            <h1>Feedback Survey</h1>
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" placeholder="Enter your name" required/>

            <label for="email">Email:</label>
            <input type="text" id="email" name="email" placeholder="Enter your email" required/>

            <label for="sherd-name">Please provide the name of the ceramic sherd you submitted:</label>
            <input type="text" id="sherd-name" name="sherd-name" placeholder="Sherd name" required/>

            <label for="sherd-portion">Which portion of the sherd did you submit?</label>
            <select id="sherd-portion" name="sherd-portion" required>
              <option value="">Select</option>
              <option value="Border">Border</option>
              <option value="Maker's Mark">Maker's Mark</option>
              <option value="Center Design">Center Design</option>
              <option value="Other">Other</option>
            </select>
            <textarea id="other-description" name="other-description" rows="2" placeholder="If 'Other', please describe..." style="display: none;"></textarea>

          
            <label for="pattern-found">Did you find the pattern you were looking for in the results? What number is it?</label>
            <input type="text" id="pattern-found" name="pattern-found" placeholder="Enter pattern number" required/>

          
            <label for="general-feedback">General Feedback:</label>
            <textarea id="general-feedback" name="general-feedback" rows="4" placeholder="Enter your feedback" required></textarea>

            <button onclick="submitSurvey()">Submit</button>
          
        </div>
      </div>
    </div>
  );
};

 
export default MatchResults;

