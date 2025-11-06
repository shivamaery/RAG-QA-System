import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import InputBox from "./InputBox";
import "./App.css";

import logo from "./assets/logo.png";

function Home() {
  return (
    <div className="home">
      <h1 className="home-header">Welcome to ResearchAI</h1>
      <p className="home-paragraph">A Small Language Model Powered Research Assistant for the McGill Computational Electromagnetics Research Group.</p>
        <div className="info-section">
          <h2>What can ResearchAI do?</h2>
          <ul>
            <li>Explain academic concepts tied to previous research.</li>
            <li>Answer questions from multiple sources.</li>
            <li>Give a concise and smart answer to your queries.</li>
          </ul>
        </div>
        <div>
          <Link to="/chat"><button className="home-chat-button">Try Asking a Question ↗</button></Link>
        </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div>
        <nav className="navbar">
          <div className="navbar-left">
            <h1 className="navbar-title">ResearchAI</h1>
          </div>

          <div className="navbar-center">
            <img src={logo} alt="McGill Logo" className="logo" />
          </div>

          <div className="navbar-right">
            <Link to="/" className="nav-link">
              Home
            </Link>
            <Link to="/chat" className="nav-link">
              Chat
            </Link>
          </div>
        </nav>

        <Routes>
          <Route
            path="/"
            element={
              <div className="page-content">
                <Home />
              </div>
            }
          />
          <Route
            path="/chat"
            element={
              <div className="page-content">
                <InputBox />
              </div>
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;