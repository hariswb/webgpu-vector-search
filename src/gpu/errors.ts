export class ErrorWebGPUInit extends Error {
    constructor(message: string) {
        super(message); 
        this.name = "ErrorWebGPUInit";
        Object.setPrototypeOf(this, ErrorWebGPUInit.prototype);
    }
}

export class ErrorWebGPUBuffer extends Error {
    constructor(message: string) {
        super(message); 
        this.name = "ErrorWebGPUBuffer";
        Object.setPrototypeOf(this, ErrorWebGPUBuffer.prototype);
    }
}

export class ErrorWebGPUCompute extends Error {
    constructor(message: string) {
        super(message); 
        this.name = "ErrorWebGPUCompute";
        Object.setPrototypeOf(this, ErrorWebGPUCompute.prototype);
    }
}
