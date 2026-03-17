import { useState, useEffect } from 'react';
import DropIn from "braintree-web-drop-in-react";
import { ArrowLeft, Lock, CheckCircle2 } from 'lucide-react';

interface Props {
  lawyer: any;
  currentUser: any;
  onBack: () => void;
  onSuccess: () => void;
}

export function PaymentGateway({ lawyer, currentUser, onBack, onSuccess }: Props) {
  const [clientToken, setClientToken] = useState<string | null>(null);
  const [instance, setInstance] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [step, setStep] = useState<'details' | 'success'>('details');

  const total = lawyer.hourlyRate * 1.05; // 5% platform fee

  useEffect(() => {
    // Fetch token from Port 5001
    fetch('http://localhost:5001/api/bookings/client_token')
      .then(res => res.text())
      .then(token => setClientToken(token))
      .catch(err => console.error("Token Fetch Error:", err));
  }, []);

  const handlePayment = async () => {
    if (!instance) return;
    setIsProcessing(true);

    try {
      // 1. Get Nonce from UI
      const { nonce } = await instance.requestPaymentMethod();

      // 2. Send to Backend
      const response = await fetch('http://localhost:5001/api/bookings/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paymentMethodNonce: nonce,
          amount: total.toFixed(2),
          lawyerId: lawyer.id,
          userId: currentUser.id
        })
      });

      const result = await response.json();
      if (result.status === 'success') {
        setStep('success');
      } else {
        alert("Payment Failed: " + result.message);
      }
    } catch (err) {
      console.error("Payment Process Error:", err);
    } finally {
      setIsProcessing(false);
    }
  };

  if (step === 'success') {
    return (
      <div className="max-w-md mx-auto bg-white p-10 rounded-3xl shadow-2xl text-center border">
        <CheckCircle2 className="w-20 h-20 text-green-500 mx-auto mb-6" />
        <h2 className="text-3xl font-bold mb-2">Payment Verified!</h2>
        <p className="text-gray-500 mb-8">Your session with {lawyer.name} is booked.</p>
        <button onClick={onSuccess} className="w-full py-4 bg-blue-600 text-white rounded-xl font-bold text-lg shadow-lg">Go to My Bookings</button>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <button onClick={onBack} className="flex items-center gap-2 text-gray-500 hover:text-blue-600 transition-all font-medium">
        <ArrowLeft className="w-4 h-4"/> Back to lawyer profile
      </button>

      <div className="bg-white p-8 rounded-3xl shadow-2xl border border-gray-100">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Secure Payment</h2>
        <p className="text-gray-500 mb-8">Supported: Credit Card, PayPal, and Google Pay</p>
        
        {clientToken ? (
          <div className="space-y-6">
            {/* Full Braintree Drop-in Integration */}
            <DropIn
              options={{ 
                authorization: clientToken,
                paypal: { flow: 'vault' },
                googlePay: {
                  merchantId: 'merchant-id-sandbox',
                  transactionInfo: {
                    totalPriceStatus: 'FINAL',
                    totalPrice: total.toFixed(2),
                    currencyCode: 'USD'
                  }
                },
                card: { cardholderName: { required: true } }
              }}
              onInstance={(inst) => setInstance(inst)}
            />

            <div className="pt-8 border-t border-gray-100 mt-4">
              <div className="flex justify-between items-center mb-8">
                <span className="text-gray-600 font-semibold text-lg">Total Charge</span>
                <span className="text-3xl font-black text-blue-600">${total.toFixed(2)}</span>
              </div>
              
              <button
                onClick={handlePayment}
                disabled={isProcessing}
                className="w-full py-5 bg-blue-600 text-white rounded-2xl font-black text-xl hover:bg-blue-700 shadow-xl shadow-blue-200 disabled:bg-gray-300 transition-all active:scale-95"
              >
                {isProcessing ? "Processing..." : "Pay Now"}
              </button>
              
              <div className="mt-6 flex items-center justify-center gap-3 text-gray-400 text-xs font-medium uppercase tracking-widest">
                <Lock className="w-4 h-4" /> PCI-DSS Compliant Gateway
              </div>
            </div>
          </div>
        ) : (
          <div className="py-24 flex flex-col items-center justify-center space-y-6">
            <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-gray-500 font-bold animate-pulse">Establishing Secure Connection...</p>
          </div>
        )}
      </div>
    </div>
  );
}