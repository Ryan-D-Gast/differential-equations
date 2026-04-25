find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.method(method.clone())/.method(method)/g' {} +
