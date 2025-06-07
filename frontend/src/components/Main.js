import React, { useState } from "react";
import axios from "axios";
import '../components/Main.css';

const Main = () => {
  const [file, setFile] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [skillsGap, setSkillsGap] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Handle file selection
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setError(""); // Reset error when a new file is selected
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setError("Please select a resume first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError("");
    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze_resume", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(response.data)
      setJobs(response.data.recommended_jobs);
      setSkillsGap(response.data.skill_gaps);
    } catch (error) {
      console.error("Error uploading resume:", error);
      setError("Failed to process the resume. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white shadow-lg rounded-lg">
      {/* <h2 className="text-xl font-semibold text-center mb-4"></h2> */}

      <form className="flex flex-col items-center gap-4">
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={handleFileChange}
          className="border p-2 w-full rounded"
        />
        <button
          type="button"
          onClick={handleUpload}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition"
        >
          {loading ? "Processing..." : "Upload & Analyze"}
        </button>
      </form>

      {error && <p className="text-red-500 mt-2">{error}</p>}

      {/* {jobs.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold">Recommended Jobs:</h3>
          <ul className="list-disc ml-4">
            {jobs.map((job, index) => (
              <li key={index} className="text-gray-700">{job}</li>
            ))}
          </ul>

          <h3 className="text-lg font-semibold mt-4">Skill Gaps:</h3>
          <ul className="list-disc ml-4">
            {skillsGap.map((skill, index) => (
              <li key={index} className="text-gray-700">{skill}</li>
            ))}
          </ul>
        </div>
      )} */}

      {jobs.length > 0 && (
        <div className="display_contents">
          {/* Recommended Jobs */}
          <div className="display_recommended_jobs">
            <h3>Recommended Jobs</h3>
            <ul className="list-disc ml-4">
              {jobs.map((job, index) => (
                <li key={index} className="text-gray-700">{job}</li>
              ))}
            </ul>
          </div>

          {/* Skill Gaps */}
          <div className="display_recommended_skills">
            <h3 className="">Skill Gaps</h3>
            <ul className="list-disc ml-4">
              {skillsGap.map((skill, index) => (
                <li key={index} className="text-gray-700">{skill}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default Main;