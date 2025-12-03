import { motion } from "framer-motion";

export default function LoadingCircle() {
  return (
    <div className="flex items-center justify-center p-4">
      <motion.div
        className="w-8 h-8 border-4 border-gray-400 border-t-transparent rounded-full"
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
      />
    </div>
  );
}