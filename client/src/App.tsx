import { useState, useEffect } from "react";
import { ChatBot } from "./components/ChatBot";
import { LawyerRecommendation } from "./components/LawyerRecommendation";
import { LawyerDirectory } from "./components/LawyerDirectory";
import { LawyerProfile } from "./components/LawyerProfile";
import { PaymentGateway } from "./components/PaymentGateway";
import { SignIn } from "./components/SignIn";
import { SignUp } from "./components/SignUp";
import { Scale, MessageSquare, Users, Home, LogIn, LogOut, User as UserIcon, CalendarCheck } from "lucide-react";
import { fetchUserProfile } from "./api";

// 1. Set your backend port to 5001 as per your terminal logs
const API_BASE = "http://localhost:5001/api";

type View = "home" | "chat" | "recommend" | "directory" | "profile" | "payment" | "signin" | "signup" | "appointments";

export interface Lawyer {
  id: string;
  name: string;
  specialty: string;
  experience: number;
  rating: number;
  hourlyRate: number;
  location: string;
  bio: string;
  cases: number;
  successRate: number;
  education: string;
  languages: string[];
  availability: string;
}

interface UserProfile {
  id: string;
  name: string;
  email: string;
}

export default function App() {
  const [currentView, setCurrentView] = useState<View>("home");
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [userBookings, setUserBookings] = useState([]);

  // Restore Session
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        if (token) {
          const user = await fetchUserProfile();
          setCurrentUser(user);
        } else if (currentView !== "signup") {
          setCurrentView("signin");
        }
      } catch (error) {
        console.error('Failed to restore session:', error);
        localStorage.removeItem('token');
        setCurrentView("signin");
      }
    };
    checkAuth();
  }, []);

  // 2. Fetch Bookings from MongoDB Atlas
  useEffect(() => {
    const fetchBookings = async () => {
      if (!currentUser || currentView !== "appointments") return;
      try {
        const res = await fetch(`${API_BASE}/bookings/user-bookings/${currentUser.id}`);
        const data = await res.json();
        setUserBookings(data);
      } catch (error) {
        console.error("Failed to fetch bookings:", error);
      }
    };
    fetchBookings();
  }, [currentView, currentUser]);

  const handleSelectLawyer = (lawyer: Lawyer) => {
    setSelectedLawyer(lawyer);
    setCurrentView("profile");
  };

  /**
   * 3. Corrected Booking Logic
   * This now opens the Payment Gateway instead of bypassing it
   */
  const handleBookLawyer = (lawyer: Lawyer) => {
    if (!currentUser) {
      setCurrentView("signin");
      return;
    }
    setSelectedLawyer(lawyer);
    setCurrentView("payment"); // Triggers the screen switch to PaymentGateway
  };

  const handleLoginSuccess = (user: UserProfile) => {
    setCurrentUser(user);
    setCurrentView("home");
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setCurrentUser(null);
    setCurrentView("home");
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 cursor-pointer" onClick={() => setCurrentView("home")}>
              <Scale className="w-8 h-8 text-blue-600" />
              <h1 className="text-blue-600 font-bold text-xl">LegalEdu Pro</h1>
            </div>
            <nav className="flex gap-2 items-center">
              {currentUser && (
                <>
                  <button onClick={() => setCurrentView("home")} className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${currentView === "home" ? "bg-blue-600 text-white" : "text-gray-700 hover:bg-gray-100"}`}><Home className="w-4 h-4" /> Home</button>
                  <button onClick={() => setCurrentView("directory")} className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${currentView === "directory" ? "bg-blue-600 text-white" : "text-gray-700 hover:bg-gray-100"}`}><Users className="w-4 h-4" /> Find Lawyers</button>
                  <button onClick={() => setCurrentView("appointments")} className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${currentView === "appointments" ? "bg-blue-600 text-white" : "text-gray-700 hover:bg-gray-100"}`}><CalendarCheck className="w-4 h-4" /> My Bookings</button>
                  <div className="w-px h-6 bg-gray-300 mx-2"></div>
                </>
              )}

              {currentUser ? (
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2 text-gray-700">
                    <UserIcon className="w-4 h-4" />
                    <span className="font-medium">{currentUser.name}</span>
                  </div>
                  <button onClick={handleLogout} className="text-red-600 hover:bg-red-50 px-4 py-2 rounded-lg"><LogOut className="w-4 h-4" /> Sign Out</button>
                </div>
              ) : (
                <button onClick={() => setCurrentView("signin")} className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center gap-2"><LogIn className="w-4 h-4" /> Sign In</button>
              )}
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {currentView === "home" && (
           <div className="grid md:grid-cols-3 gap-6 pt-10">
              <button onClick={() => setCurrentView("recommend")} className="bg-white p-8 rounded-xl shadow-sm border hover:shadow-md transition-all text-left">
                <Scale className="w-12 h-12 text-blue-600 mb-4" />
                <h3 className="font-bold text-xl mb-2">Lawyer Match</h3>
                <p className="text-gray-600">Get recommendations based on your case.</p>
              </button>
              <button onClick={() => setCurrentView("directory")} className="bg-white p-8 rounded-xl shadow-sm border hover:shadow-md transition-all text-left">
                <Users className="w-12 h-12 text-blue-600 mb-4" />
                <h3 className="font-bold text-xl mb-2">Browse All</h3>
                <p className="text-gray-600">Explore our directory of experts.</p>
              </button>
              <button onClick={() => setCurrentView("chat")} className="bg-white p-8 rounded-xl shadow-sm border hover:shadow-md transition-all text-left">
                <MessageSquare className="w-12 h-12 text-blue-600 mb-4" />
                <h3 className="font-bold text-xl mb-2">Legal AI</h3>
                <p className="text-gray-600">Ask basic legal questions to our AI.</p>
              </button>
           </div>
        )}

        {currentView === "directory" && <LawyerDirectory onSelectLawyer={handleSelectLawyer} onGoToRecommendations={() => setCurrentView("recommend")} />}
        {currentView === "recommend" && <LawyerRecommendation onSelectLawyer={handleSelectLawyer} />}
        {currentView === "chat" && <ChatBot />}
        
        {currentView === "profile" && selectedLawyer && (
          <LawyerProfile lawyer={selectedLawyer} onBook={handleBookLawyer} onBack={() => setCurrentView("directory")} />
        )}

        {/* 4. Payment Gateway Route with currentUser passed */}
        {currentView === "payment" && selectedLawyer && (
          <PaymentGateway 
            lawyer={selectedLawyer} 
            currentUser={currentUser} 
            onBack={() => setCurrentView("profile")} 
            onSuccess={() => setCurrentView("appointments")} 
          />
        )}

        {/* 5. Booking History View */}
        {currentView === "appointments" && (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-gray-900">My Confirmed Bookings</h2>
            <div className="grid gap-4">
              {userBookings.length === 0 ? (
                <p className="bg-white p-10 rounded-xl border border-dashed text-center text-gray-500">No appointments found.</p>
              ) : (
                userBookings.map((b: any) => (
                  <div key={b._id} className="bg-white p-6 rounded-xl shadow-sm border flex justify-between items-center">
                    <div className="flex items-center gap-4">
                       <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold text-xl">
                          {b.status === 'confirmed' ? '✓' : '?'}
                       </div>
                       <div>
                          <p className="font-bold text-lg">Legal Consultation</p>
                          <p className="text-sm text-gray-500">Order ID: {b.razorpayOrderId || b.transactionId}</p>
                       </div>
                    </div>
                    <div className="text-right">
                       <p className="font-bold text-xl text-blue-600">${b.amount}</p>
                       <span className="text-xs font-bold uppercase text-green-600 bg-green-50 px-2 py-1 rounded-full">Confirmed</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {currentView === "signin" && <SignIn onSuccess={handleLoginSuccess} onNavigateToSignUp={() => setCurrentView("signup")} />}
        {currentView === "signup" && <SignUp onSuccess={handleLoginSuccess} onNavigateToSignIn={() => setCurrentView("signin")} />}
      </main>
    </div>
  );
}