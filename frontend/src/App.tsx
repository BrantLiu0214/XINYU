import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { NavBar } from './components/NavBar';
import { RequireAuth } from './components/RequireAuth';
import { ChatPage } from './pages/ChatPage';
import { DashboardPage } from './pages/DashboardPage';
import { SessionDetailPage } from './pages/SessionDetailPage';

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route
          path="/dashboard"
          element={<RequireAuth><DashboardPage /></RequireAuth>}
        />
        <Route
          path="/session/:sessionId"
          element={<RequireAuth><SessionDetailPage /></RequireAuth>}
        />
      </Routes>
    </BrowserRouter>
  );
}
